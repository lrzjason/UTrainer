# 双联图（IC-LoRA）编辑标注规则

> 适用于 `E:\images\MOSS_IC_CONSISTENCY`（左 = 原图 / 右 = 编辑后，同一画布并排）。
> 实现：`UnifiedTrainer/scripts/build_diptych_annotations.py`（本文件即其规格说明）。

```bash
python -m UnifiedTrainer.scripts.build_diptych_annotations \
  --data E:\images\MOSS_IC_CONSISTENCY\train_flat \
  --out  E:\images\MOSS_IC_CONSISTENCY\v10
```

---

## 1. 编辑区域：**测量**，不靠手标

| 步骤 | 参数 | 说明 |
|---|---|---|
| 1. 去噪 | 高斯 σ=1.2 | 左右两半各模糊，压掉 JPEG/重采样噪声 |
| 2. 求差 | 逐像素**最大通道**差的绝对值 | 比亮度差更敏感于色相变化 |
| 3. 阈值 | `> 30` | 低于 30 的多为压缩噪声 |
| 4. 忽略边带 | 距面板左右边界 **10 px** 内置零 | 两半不是逐像素对齐，接缝必然有差异 |
| 5. 连通域过滤 | 面积 **≥ 500 px** | 更小的碎片是噪声 |
| 6. 接缝剔除 | 宽≤2px 且高>40px；或宽<8px 且高>30%画布；或贴边 6px 且高>50%画布 | 竖条纹、面板边框 |
| 7. 开运算 | 3×3 | 去掉 1–2 px 细线 |
| 8. 外接框 | 剩余**全部**差异像素的 bounding box | **不取最大块，也不按块拆** |
| 9. 外扩 | `max(10 px, 画布 1%)`，并夹在右半幅内 | 让框完整包住红色标注范围 |
| 10. 坐标换算 | 图像像素 → 画布坐标（`W/iw`, `H/ih`） | 画布尺寸取 JSON 的 `width/height` |

> 与早期版本的区别：早期用"最大连通块"（会漏掉分散改动）或"全部块并集但不外扩"（框贴太紧）；
> 现在统一为 **并集 + 外扩**。

## 2. 编辑项：**恰好一个**，带 `operation`

1. 在**右半幅**的 item 中找与测量框 IoU 最大者；
2. **IoU ≥ 0.5** → 直接**调整**该 item 的 `bbox` 为测量框（caption 加 `(edited area)`，继承原 `label`/`text`）——
   因为"含义与位置本来就一致"，不应重复添加；
3. 否则**追加**一个新 item（`label` 取最近 item 的标签，`text` 一并带上）；
4. 无论哪种方式，该 item 带 **`operation: "edit"`**，全组唯一。

## 3. 其余 item：**全部保留**

原始标注的每个 item 原样写入（`label` / `caption` / `bbox` / `text`）。
`compile_prompt` 只认识这 4 个字段，`operation` 在**编译时剥离**，不污染官方 prompt 格式。

## 4. 左右镜像：**同一物体必须映射相等**

右半 = 左半整体右移 `width // 2`，因此同一物体的框必须满足：

```
right_box == left_box + (width // 2, 0)      # 同宽同高、同 y
```

- 按 caption（去 `kept unchanged…`、去 `(edited area)`、小写）配对；
- 配不到再按几何最近（`left+half` 距离 < 12 px）配对；
- 不一致时**以左半为基准**改写右半；
- **编辑项豁免**（它的差异是设计本身）。

## 4.1 编辑的"改动前"项：`operation: "edit_before"`

一次编辑天然有 before / after 两个状态，二者落在**同一位置**（相对各自面板）：

```
before(left) = edit_box - (width // 2, 0)      # 左半幅的"改之前"
after(right) = edit_box                        # 右半幅的"改之后"
```

判定：左半幅的某个 item 满足
1. 没有右半镜像对（`right = left + half` 找不到）；
2. 它的镜像框与编辑框 **IoU ≥ 0.5**；

则标 `operation: "edit_before"`，caption 追加 `(before the edit)`。
当前 37 组里 **19 组**命中（例：039 左 `Text "OPEN"` ↔ 右 `EDIT` 的 `GALLERY`；
037 左 `A horse's head` ↔ 右 `A tiger's head`）。

预览图里：橙框 `BEFORE #id` = 改动前，红框 `EDIT #id` = 改动后。

## 5. 剔除规则

| 类型 | 判定 |
|---|---|
| 人工复核剔除 | `DROP` 集合（当前 8 组） |
| 整面板重绘 | diff 覆盖过大（早期用 8–10% 框面积阈值；现改为人工确认后加入 `DROP`） |

当前保留 **37 / 50** 组。

## 6. 输出

| 文件 | 内容 |
|---|---|
| `annotations/<stem>.json` | 草稿布局：`width/height` + `bbox_coord_space: "abs"` + `bbox_format: "xyxy"` + `summary` + `items`（编辑项带 `operation`） |
| `captions/<stem>.txt` | **编译后**的官方 prompt：一行 JSON，`abs_centralized` / `yxyx` |
| `bbox/<stem>.jpg` | 审查图：每个 item 编号，编辑框**粗红**并标 `EDIT #<id>`，左右分界白线 |

## 7. 编号与可视化约定

- 编号 = 该 item 在 `items` 数组中的 **1-based 位置**（也是 caption 里的顺序）；
- 编辑框标 `EDIT #<id>`，其余标数字；
- 黄框 = 左半 item，青框 = 右半 `kept unchanged` item，红框 = 编辑项。

## 8. 已知遗留问题

1. **041** 的 `GOAL` / `ENERGY DRINK` 文字框位置与画面内容不符（左右一致但整体偏），需按图像内容重定位；
2. 19 组各有 1 个 `edit_before` 项（左半幅"改之前"），已标注而非当作噪声；这 19 组的该位置在右半幅没有镜像对，属编辑的 before 状态；
3. 7 组编辑框占画布 21–35%（diff 本身分散），如需更紧可改为"每个连通块一个框"。
