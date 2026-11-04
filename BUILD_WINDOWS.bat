@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Instale o Python 3 para Windows antes de criar o executável.
  goto :error
)

py -3 -c "import PyInstaller, cv2, mediapipe, numpy, pyvirtualcam, pygame" >nul 2>nul
if errorlevel 1 (
  echo Instalando as dependências de compilação...
  py -3 -m pip install -r requirements.txt "pyinstaller>=6,<7"
  if errorlevel 1 goto :error
)

if not exist "models\gesture_recognizer.task" goto :models
if not exist "models\selfie_segmenter_landscape.tflite" goto :models
goto :build

:models
echo Baixando os modelos necessários do MediaPipe...
py -3 -c "from vanish_cam import download_models; download_models()"
if errorlevel 1 goto :error

:build
py -3 -m PyInstaller --noconfirm --distpath dist --workpath build VanishCam.spec
if errorlevel 1 goto :error

"dist\VanishCam.exe" --self-test
if errorlevel 1 goto :error

echo.
echo Concluído: dist\VanishCam.exe
exit /b 0

:error
echo.
echo A compilação falhou. Confira as mensagens acima.
pause
exit /b 1
