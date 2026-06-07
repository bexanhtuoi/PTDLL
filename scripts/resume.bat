@echo off
set PYTHONPATH=C:\Users\PC\Downloads\PTDLL\src
cd /d C:\Users\PC\Downloads\PTDLL
uv run python -m src.main portfolio train --mode resume --models sac td3 %*
