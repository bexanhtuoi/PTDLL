@echo off
set PYTHONPATH=C:\Users\PC\Downloads\PTDLL\src
cd /d C:\Users\PC\Downloads\PTDLL
if "%*"=="" (
    uv run python -m src.main portfolio train
) else (
    uv run python -m src.main %*
)
