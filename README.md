# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Download for Windows

[Download VanishCam.exe](https://nicoaauriemma.vercel.app/shop) and double-click it to run the app. You do not need to install Python.

## Getting started

1. Double-click `AVVIA_LOCALE.bat`. On the first run it creates `.venv-local` and installs the dependencies, so internet is needed only once.
2. In the start menu, choose **Normal Vanish**, **Naruto Vanish**, or **Goku Teleport** with the mouse or the `1`, `2`, and `3` keys.
3. Step out of the camera frame for 4 seconds while the background is captured. The terminal tells you when you can step back in.

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 3.6 seconds.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

**Goku Teleport:** Hold your index and middle fingers together against your forehead for about one second. The body disappears behind vertical comic speed lines, which linger briefly over the silhouette. Repeat the same gesture to come back. Goku mode uses `hand_landmarker.task` and `face_landmarker.task` to locate the fingers and the forehead. `assets/teleport.MP3` plays on both transitions when present; otherwise the app synthesizes a short teleport sound. `V` and `R` are disabled in this mode, and a countdown is printed only in the terminal.

To start a mode without the menu:

```powershell
AVVIA_LOCALE.bat --mode naruto
AVVIA_LOCALE.bat --mode normal
AVVIA_LOCALE.bat --mode goku
```

The MediaPipe models in `models/` are downloaded automatically on the first run. To check that models, assets, and audio are present without opening the webcam, run:

```powershell
.venv-local\Scripts\python.exe vanish_cam.py --self-test
```

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `AVVIA_VIRTUAL.bat` to require a virtual camera. Select **OBS Virtual Camera** in the app that will receive the video. OBS's virtual camera must be stopped while this app is producing its output.

To use the effect in an OBS scene, run `AVVIA_OBS.bat`, add a **Window Capture** source, select **Webcam - Anteprima**, and then start the virtual camera from OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
AVVIA_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

- `SPACE`: capture a new background.
- `R`: make the person reappear (Naruto and Normal modes).
- `V`: manually start the selected mode's effect (Naruto and Normal modes).
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit and release the cameras.

You can change the camera, resolution, and maximum virtual-output FPS:

```powershell
AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Running from source

`AVVIA_LOCALE.bat` looks for Python 3.10 or newer (preferring 3.12/3.13) and creates the local environment automatically. On Python 3.14 the script installs `pygame-ce`, which provides the same `pygame` module used by the app.

To install the dependencies manually:

```powershell
py -3 -m venv .venv-local
.venv-local\Scripts\python.exe -m pip install -r requirements.txt
.venv-local\Scripts\python.exe vanish_cam.py
```

To run the tests:

```powershell
.venv-local\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. The Goku speed lines are generated in real time, and `vanish_sound.py` synthesizes a teleport WAV when `assets/teleport.MP3` is not provided. No anime frames or assets are included in the project.
