@echo off
REM This script installs the usbmd package and its dependencies.
REM Author: Tristan Stevens
REM Date: 2023-09-25

REM For windows only! Note: this is not a general purpose installation script for usbmd as every machine is different.
REM simple pip install usbmd should work for most users.

REM Example usage: custom_installation.bat "D:\Username\Projects\Ultrasound-BMd" "D:\Username\conda\envs\usbmd"
REM if you specify the env_path, the last folder will also be the env name.

REM Parse command line arguments
set "repo_root=%~1"
set "env_path=%~2"

REM Check if Conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda is not installed. Please install Conda first.
    exit /b 1
)
echo Conda already installed

REM Set default value for repo root if not provided
if "%repo_root%" == "" (
    set "repo_root=D:\usbmd\Ultrasound-BMd"
)
echo Repo root: %repo_root%
echo Environment path: %env_path%

REM Set default value for environment name if not provided
call conda create --prefix %env_path% -y

call activate %env_path%

call conda install python=3.9 -y
call python -m pip install --upgrade pip

REM Install usbmd
REM which runs the following under the hood as well:
REM pip install -r requirements.txt
cd "%repo_root%"
call python -m pip install -e .

call conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

call python -m pip install "tensorflow<2.11"
REM Verify install:
call python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

call conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia -y
call conda install cudatoolkit -y
REM Verify install:
call python -c "import torch; print(torch.cuda.is_available())"

PAUSE
