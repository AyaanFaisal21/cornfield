@echo off
REM Run a Python command inside the MSVC env (for PyTorch CUDA extension builds),
REM reusing TransformerOp's venv (torch + ninja + CUDA toolchain already installed).
REM   Usage (from this dir):  winbuild.bat -m autotune_matmul
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VENV=D:\Users\ayaan faisal\Downloads\ps1\TransformerOp\.venv"
cd /d "%~dp0"
set "PATH=%VENV%\Scripts;%PATH%"
"%VENV%\Scripts\python.exe" %*
