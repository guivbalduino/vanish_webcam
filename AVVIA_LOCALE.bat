@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv-local"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [VanishCam] Criando o ambiente Python local...
  set "BASE_PY="
  if not defined BASE_PY (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BASE_PY=py -3.12"
  )
  if not defined BASE_PY (
    py -3.13 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BASE_PY=py -3.13"
  )
  if not defined BASE_PY (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BASE_PY=py -3.11"
  )
  if not defined BASE_PY (
    py -3.10 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BASE_PY=py -3.10"
  )
  if not defined BASE_PY (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BASE_PY=py -3"
  )
  if not defined BASE_PY (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BASE_PY=python"
  )
  if not defined BASE_PY (
    echo [VanishCam] Python 3 não encontrado. Instale o Python 3.10 ou mais recente.
    goto :error
  )
  %BASE_PY% -m venv "%VENV_DIR%"
  if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import cv2, mediapipe, numpy, pyvirtualcam, pygame" >nul 2>nul
if errorlevel 1 (
  echo [VanishCam] Instalando os componentes necessários. A internet será usada apenas nesta etapa...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%VENV_PY%" -m pip install -r "requirements.txt"
  if errorlevel 1 goto :error
)

echo [VanishCam] Iniciando a webcam...
"%VENV_PY%" "vanish_cam.py" %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Não foi possível iniciar o VanishCam. Consulte o README.md para ver soluções comuns.
pause
exit /b 1
