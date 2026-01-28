@echo off
setlocal
cd /d "%~dp0"

set CELLPOSE_LOCAL_MODELS_PATH=%~dp0cellpose_weights

call C:\Users\claud\miniconda3\Scripts\activate.bat biomarker_mvp
python "%~dp0qt_mvp_app_v3.py"
pause