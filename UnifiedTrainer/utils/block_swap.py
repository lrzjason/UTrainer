"""
Block swap offloading utilities.

Copied from kohya-ss/sd-scripts (custom_offloading_utils.py) and adapted
for UnifiedTrainer.  Enables swapping transformer blocks between CPU and
GPU during forward/backward passes to reduce peak VRAM usage.

Usage:
    from UnifiedTrainer.utils.block_swap import ModelOffloader
    offloader = ModelOffloader(blocks, num_blocks, blocks_to_swap, device)
    offloader.prepare_block_devices_before_forward(blocks)
    # In forward loop:
    offloader.wait_for_block(i)
    ... block forward ...
    offloader.submit_move_blocks(blocks, i)
"""
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional
import torch
import torch.nn as nn


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _flush():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_frozen_weight(module):
    """Check if a module's weight is frozen (not trainable).
    LoRA adapter weights (requires_grad=True) should never be swapped to CPU.
    """
    return not (hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad)


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            # Skip trainable weights (LoRA params) -they must stay on GPU
            if module_to_cuda.weight.requires_grad:
                continue
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append(
                    (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                )
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    modules_cpu = list(layer_to_cpu.modules())
    modules_cuda = list(layer_to_cuda.modules())
    for module_to_cpu, module_to_cuda in zip(modules_cpu, modules_cuda):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            # Skip trainable weights (LoRA params) -they must stay on GPU
            if module_to_cpu.weight.requires_grad:
                continue
            weight_swap_jobs.append(
                (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
            )

    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device(device)

    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device(device)


def weights_to_device(layer: nn.Module, device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            # Skip trainable weights (LoRA params) -they must stay on GPU
            if module.weight.requires_grad:
                continue
            module.weight.data = module.weight.data.to(device, non_blocking=True)


class Offloader:
    """Common offloading base class."""

    def __init__(self, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(f"Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to device")

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"Moved blocks in {time.perf_counter()-start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"


class ModelOffloader(Offloader):
    """Supports forward + backward offloading with hooks."""

    def __init__(self, blocks, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        super().__init__(num_blocks, blocks_to_swap, device, debug)

        self.remove_handles = []
        for i, block in enumerate(blocks):
            hook = self.create_backward_hook(blocks, i)
            if hook is not None:
                handle = block.register_full_backward_hook(hook)
                self.remove_handles.append(handle)

    def __del__(self):
        for handle in self.remove_handles:
            handle.remove()

    def create_backward_hook(self, blocks, block_index: int) -> Optional[callable]:
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        for b in blocks[0: self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weights_to_device(b, self.device)

        for b in blocks[self.num_blocks - self.blocks_to_swap:]:
            b.to(self.device)
            weights_to_device(b, "cpu")

        synchronize_device(self.device)
        _flush()

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks(self, blocks, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if block_idx >= self.blocks_to_swap:
            return
        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
