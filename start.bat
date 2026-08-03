@echo off
REM UnifiedTrainer Orchestrator — Windows start script
cd /d "%~dp0"
python -m orchestrator.main --workspace workspace --api --port 7860 --max-parallel 2
