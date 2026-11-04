"""Webcam locale: dissolvenza avviata dal gesto con indice e medio a V."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from .vanish_sound import (
        SOUND_PATH, TeleportSound, VanishSound, teleport_audio_path,
    )
except ImportError:
    from naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from vanish_sound import (
        SOUND_PATH, TeleportSound, VanishSound, teleport_audio_path,
    )


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Pr\u00e9via"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = APP_DIR / "face_landmarker.task"
SAMPLE_TEST_IMAGE = APP_DIR / "test_assets" / "sample_person.png"
DEFAULT_TEST_OUTPUT = Path.cwd() / "test-output" / "sample_person_preview.png"

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 3.6
GESTURE_INTERVAL_SECONDS = 0.14
GOKU_INTERVAL_SECONDS = 0.08
GOKU_HOLD_SECONDS = 1.0
GOKU_GESTURE_THRESHOLD = 0.62
GOKU_EFFECT_SECONDS = 0.95
GOKU_LINGER_SECONDS = 0.45

_VALID_EFFECTS = {"normal", "naruto", "goku"}


class BrazilianPortugueseHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        prefix = (prefix or "uso: ").replace("usage:", "uso:")
        return super()._format_usage(usage, actions, groups, prefix)


class BrazilianPortugueseArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        message = message.replace("unrecognized arguments:", "argumentos n\u00e3o reconhecidos:")
        message = message.replace("invalid choice:", "op\u00e7\u00e3o inv\u00e1lida:")
        message = message.replace("(choose from", "(op\u00e7\u00f5es:")
        message = message.replace("invalid int value:", "n\u00famero inteiro inv\u00e1lido:")
        message = message.replace("invalid float value:", "n\u00famero decimal inv\u00e1lido:")
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: erro: {message}\n")


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    phase: str = "calibrating"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False
    goku_hold_started: float | None = None
    goku_absent_since: float | None = None
    goku_last_pose: float | None = None
    goku_armed: bool = True
    goku_countdown: int | None = None
    active_vanish: str | None = None
    goku_linger_started: float = 0.0


@dataclass
class EffectState:
    """Per-session effect state kept separately from calibration/fullscreen."""
    phase: str = "ready"
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    active_vanish: str | None = None
    # goku-specific
    goku_hold_started: float | None = None
    goku_absent_since: float | None = None
    goku_last_pose: float | None = None
    goku_armed: bool = True
    goku_countdown: int | None = None
    goku_linger_started: float = 0.0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("A taxa de quadros (FPS) deve ficar entre 1 e 120.")
    return fps


def _parse_effects(raw: str) -> set[str]:
    """Parse a comma-separated effects string and validate each token."""
    effects: set[str] = set()
    for token in raw.split(","):
        token = token.strip().lower()
        if token not in _VALID_EFFECTS:
            raise argparse.ArgumentTypeError(
                f"Efeito n\u00e3o reconhecido: '{token}'. Escolha entre: {', '.join(sorted(_VALID_EFFECTS))}"
            )
        effects.add(token)
    return effects


def parse_args() -> argparse.Namespace:
    parser = BrazilianPortugueseArgumentParser(
        description="Escolha um efeito Vanish ou uma combina\u00e7\u00e3o para a webcam.",
        add_help=False,
        formatter_class=BrazilianPortugueseHelpFormatter,
    )
    parser._positionals.title = "argumentos posicionais"
    parser._optionals.title = "op\u00e7\u00f5es"
    parser.add_argument("-h", "--help", action="help", help="Exibe esta ajuda e encerra")
    parser.add_argument("--camera", type=int, default=0, help="\u00cdndice da webcam")
    parser.add_argument("--width", type=int, default=960, help="Largura do v\u00eddeo")
    parser.add_argument("--height", type=int, default=540, help="Altura do v\u00eddeo")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Inicia em tela cheia"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="N\u00e3o espelhar a imagem da webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Limite de FPS da c\u00e2mera virtual (padr\u00e3o: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Exige a c\u00e2mera virtual; por padr\u00e3o, ela \u00e9 ativada se estiver dispon\u00edvel. --no-virtual-camera a desativa",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Driver da c\u00e2mera virtual",
    )
    parser.add_argument("--virtual-device", help="Nome do dispositivo virtual")
    parser.add_argument(
        "--mode", choices=("normal", "naruto", "goku", "combo"),
        help="Pula o menu e inicia diretamente o modo escolhido (combo ativa Naruto e Goku)",
    )
    parser.add_argument(
        "--effects", default=None,
        help="Lista de efeitos ativos separados por v\u00edrgula (ex.: naruto,goku)",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Exibe o estado do reconhecimento no terminal, sem texto sobre o v\u00eddeo",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--test-image", nargs="?", const=SAMPLE_TEST_IMAGE, type=Path,
        help="Processa uma imagem est\u00e1tica (por padr\u00e3o, usa a imagem de exemplo)",
    )
    parser.add_argument(
        "--test-output", type=Path, default=DEFAULT_TEST_OUTPUT,
        help="Caminho da pr\u00e9via gerada por --test-image",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Menu functions
# ---------------------------------------------------------------------------

def choose_mode() -> str | None:
    """Simple keyboard/mouse menu shown before the clean camera feed."""
    title = "Escolha um efeito"
    selected = [None]
    window = np.zeros((580, 700, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "ESCOLHA SEU EFEITO VANISH", (68, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
    buttons = (
        ("normal", "1  VANISH NORMAL", "Sinal de V / desaparecimento", 100),
        ("naruto", "2  SUBSTITUICAO NARUTO", "Maos unidas / tronco e fumaca", 210),
        ("goku", "3  TELETRANSPORTE GOKU", "Dedos na testa / teletransporte", 320),
        ("combo", "4  GOKU + NARUTO", "Os dois efeitos na mesma cena", 430),
    )
    for mode, heading, detail, top in buttons:
        cv2.rectangle(window, (65, top), (635, top + 94), (48, 54, 62), -1, cv2.LINE_AA)
        cv2.rectangle(window, (65, top), (635, top + 94), (111, 142, 153), 2, cv2.LINE_AA)
        cv2.putText(window, heading, (90, top + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.77, (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(window, detail, (90, top + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 65 <= x <= 635:
            for mode, _, _, top in buttons:
                if top <= y <= top + 94:
                    selected[0] = mode

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("1"):
                selected[0] = "normal"
            elif key == ord("2"):
                selected[0] = "naruto"
            elif key == ord("3"):
                selected[0] = "goku"
            elif key == ord("4"):
                selected[0] = "combo"
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


def choose_effects() -> set[str] | None:
    """Checkbox-based multi-effect selector. Returns a set of selected effect names."""
    title = "Escolha os efeitos"
    WIN_W, WIN_H = 700, 460
    CHECK_SIZE = 28
    INACTIVE_COL = (48, 54, 62)
    ACTIVE_COL = (31, 120, 92)
    BORDER_INACTIVE = (111, 142, 153)
    BORDER_ACTIVE = (72, 220, 160)
    TEXT_COL = (240, 245, 247)
    DETAIL_COL = (179, 195, 201)
    BTN_ACTIVE_COL = (31, 120, 92)
    BTN_INACTIVE_COL = (60, 60, 60)
    BG_COL = (26, 29, 34)

    items = [
        ("normal", "Vanish normal", "Sinal de V / desaparecimento"),
        ("naruto", "Substituicao Naruto", "Maos unidas / tronco e fumaca"),
        ("goku",   "Teletransporte Goku", "Dedos na testa / teletransporte"),
    ]
    ROW_Y = [100, 185, 270]
    BTN_Y = 370

    selected: set[str] = set()
    confirmed = [False]

    def _render() -> np.ndarray:
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:] = BG_COL
        cv2.putText(canvas, "ESCOLHA OS EFEITOS", (120, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COL, 2, cv2.LINE_AA)
        for idx, (eff, heading, detail) in enumerate(items):
            top = ROW_Y[idx]
            is_sel = eff in selected
            bg = ACTIVE_COL if is_sel else INACTIVE_COL
            bd = BORDER_ACTIVE if is_sel else BORDER_INACTIVE
            cv2.rectangle(canvas, (65, top), (635, top + 70), bg, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (65, top), (635, top + 70), bd, 2, cv2.LINE_AA)
            # Checkbox square
            cx, cy = 95, top + 35
            half = CHECK_SIZE // 2
            cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half),
                          (220, 220, 220), 2, cv2.LINE_AA)
            if is_sel:
                cv2.putText(canvas, "X", (cx - half + 5, cy + half - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (72, 220, 160), 2, cv2.LINE_AA)
            key_char = str(idx + 1)
            cv2.putText(canvas, f"[{key_char}] {heading}", (130, top + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT_COL, 2, cv2.LINE_AA)
            cv2.putText(canvas, detail, (130, top + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, DETAIL_COL, 1, cv2.LINE_AA)
        # INICIAR button
        has_sel = len(selected) >= 1
        btn_bg = BTN_ACTIVE_COL if has_sel else BTN_INACTIVE_COL
        cv2.rectangle(canvas, (200, BTN_Y), (500, BTN_Y + 55), btn_bg, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (200, BTN_Y), (500, BTN_Y + 55), (150, 150, 150), 2, cv2.LINE_AA)
        label_col = TEXT_COL if has_sel else (120, 120, 120)
        cv2.putText(canvas, "INICIAR  [ENTER]", (230, BTN_Y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, label_col, 2, cv2.LINE_AA)
        return canvas

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for idx, (eff, _, _) in enumerate(items):
            top = ROW_Y[idx]
            if 65 <= x <= 635 and top <= y <= top + 70:
                if eff in selected:
                    selected.discard(eff)
                else:
                    selected.add(eff)
                return
        if len(selected) >= 1 and 200 <= x <= 500 and BTN_Y <= y <= BTN_Y + 55:
            confirmed[0] = True

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while not confirmed[0]:
            cv2.imshow(title, _render())
            key = cv2.waitKey(30) & 0xFF
            if key == ord("1"):
                selected.symmetric_difference_update({"normal"})
            elif key == ord("2"):
                selected.symmetric_difference_update({"naruto"})
            elif key == ord("3"):
                selected.symmetric_difference_update({"goku"})
            elif key in (13, 10):  # ENTER
                if len(selected) >= 1:
                    confirmed[0] = True
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return set(selected)
    finally:
        cv2.destroyWindow(title)


# ---------------------------------------------------------------------------
# Camera / virtual camera
# ---------------------------------------------------------------------------

def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "C\u00e2mera virtual indispon\u00edvel. Instale o OBS Studio (com o driver "
            "OBS Virtual Camera) ou o Unity Capture. Se estiver usando a OBS Virtual Camera, "
            "desative Iniciar c\u00e2mera virtual no OBS. "
            "Execute AVVIA_LOCALE.bat para instalar o pyvirtualcam. "
            f"Detalhes: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Continuando com a pr\u00e9via local.")
        return None
    print(f"[Webcam] C\u00e2mera virtual ativa: {camera.device} ({width}x{height}, {args.fps:g} FPS m\u00e1ximos)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} indispon\u00edvel. Feche outros aplicativos ou tente --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


# ---------------------------------------------------------------------------
# Model downloads
# ---------------------------------------------------------------------------

def download_models(active_effects: "str | set[str]" = "normal") -> None:
    """Download required models. Accepts a mode string or a set of effect names."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Normalize to a set of effects
    if isinstance(active_effects, str):
        mode = active_effects
        if mode in ("naruto", "goku", "combo"):
            skip_gesture = True
        else:
            skip_gesture = False
    else:
        skip_gesture = "normal" not in active_effects

    for destination, url in MODEL_URLS.items():
        if skip_gesture and destination == GESTURE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[VanishCam] Baixando {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Vanish-Lab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "N\u00e3o foi poss\u00edvel baixar os modelos do MediaPipe. "
                "Verifique sua conex\u00e3o com a internet e tente novamente."
            ) from error


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], background[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8)
        + cyan.astype(np.float32) * glow * 0.8,
        0,
        255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


# ---------------------------------------------------------------------------
# Gesture recognition — Normal (Victory)
# ---------------------------------------------------------------------------

def victory_landmark_score(points: np.ndarray) -> float:
    """Riconosci indice e medio distesi usando distanze 3D, non l'orientamento."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP e punta: indice, medio, anulare, mignolo.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Di lato le dita ripiegate possono apparire dritte ma molto corte:
    # non pretendere articolazioni ben visibili per anulare e mignolo.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # Le due punte devono essere separate; il pollice puo' restare nascosto.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Qualita' geometrica delle dita distese, non probabilita' del classificatore.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # I landmark vengono restituiti anche quando il classificatore non riconosce
    # "Victory" (ad esempio con il dorso della mano rivolto alla webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # Controlla SEMPRE anche la geometria visibile: la profondita' stimata
    # puo' essere imprecisa quando palmo e dita piegate sono nascosti.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Riprova la rilevazione su fotogrammi ruotati senza alterare il tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE forza una nuova rilevazione; il modello VIDEO resta sul frame
        # originale. Evita timestamp/tracking incoerenti tra rotazioni diverse.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # Il palmo nella posa laterale e' spesso troppo piccolo nel frame
        # completo. I due ritagli inferiori si sovrappongono e coprono entrambe
        # le mani, anche quando l'immagine e' specchiata.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0
        self._last_video_timestamp: int | None = None
        self._last_video_result = None

    def detect_video(self, image: mp.Image, timestamp_ms: int):
        """Run the video tracker once per frame, sharing results across effects."""
        if timestamp_ms == self._last_video_timestamp:
            return self._last_video_result
        if (
            self._last_video_timestamp is not None
            and timestamp_ms < self._last_video_timestamp
        ):
            raise ValueError("Os registros de tempo do rastreador de m\u00e3os devem sempre avan\u00e7ar.")
        result = self.primary.detect_for_video(image, timestamp_ms)
        self._last_video_timestamp = timestamp_ms
        self._last_video_result = result
        self.last_hand_count = len(result.hand_landmarks)
        return result

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.detect_video(image, timestamp_ms)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Avvia la dissolvenza al primo riconoscimento affidabile della V."""
    if state.phase == "ready" and name == "Victory" and score >= 0.62:
        state.phase = "vanishing"
        state.vanish_started = now
        return True
    return False


def start_naruto_on_join(state: AppState, joined: bool, now: float) -> bool:
    if state.phase == "ready" and joined:
        state.phase = "naruto_vanishing"
        state.vanish_started = now
        return True
    return False


# ---------------------------------------------------------------------------
# Goku effect — gesture detection
# ---------------------------------------------------------------------------

def goku_forehead(
    face_landmarks: list,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    """Return (x, y, face_height, face_width) of the forehead in pixels, or None."""
    if len(face_landmarks) < 468:
        return None
    top = face_landmarks[10]
    bottom = face_landmarks[152]
    left = face_landmarks[234]
    right = face_landmarks[454]
    face_h = abs(bottom.y - top.y) * frame_height
    face_w = abs(right.x - left.x) * frame_width
    if face_h < 30:
        return None
    cx = top.x * frame_width
    cy = top.y * frame_height
    return cx, cy, face_h, face_w


def goku_gesture_score(
    points: np.ndarray,
    forehead: tuple[float, float, float, float] | None,
) -> float:
    """Score for two fingers (index+middle) close together on forehead."""
    if forehead is None:
        return 0.0
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    # Index and middle must be extended; ring and pinky folded.
    extended = []
    folded = []
    for idx, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_dist = float(np.linalg.norm(finger[3] - points[0]))
        joint_dist = float(np.linalg.norm(finger[1] - points[0]))
        extended.append(ratio >= 0.82 and tip_dist > joint_dist * 1.10)
        folded.append(ratio < 0.78 or tip_dist <= joint_dist * 1.05)

    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0

    # Two fingers must be close together (not spread like a V).
    sep = float(np.linalg.norm(points[8] - points[12]))
    finger_len = float(np.linalg.norm(points[5] - points[8]))
    if finger_len < 1e-6:
        return 0.0
    if sep > finger_len * 0.45:
        return 0.0

    # Fingertips must be near the forehead position.
    fx, fy, fh, fw = forehead
    tip_x = float((points[8, 0] + points[12, 0]) * 0.5)
    tip_y = float((points[8, 1] + points[12, 1]) * 0.5)
    dist = float(np.sqrt((tip_x - fx) ** 2 + (tip_y - fy) ** 2))
    if dist > fh * 0.65:
        return 0.0

    # Score based on how close the tips are to the forehead centre.
    proximity = max(0.0, 1.0 - dist / max(fh * 0.65, 1e-6))
    return float(np.clip(proximity, 0.0, 1.0))


def update_goku_hold(state: AppState, score: float, now: float) -> str | None:
    """
    Return 'vanish', 'return', or None based on how long the pose is held.
    Mutates state fields in-place.
    """
    holds = score >= GOKU_GESTURE_THRESHOLD
    phase = state.phase

    # Track absence to re-arm after vanish.
    if phase == "vanished":
        if holds:
            # Still on forehead while vanished — don't re-arm yet.
            state.goku_absent_since = None
        else:
            if state.goku_absent_since is None:
                state.goku_absent_since = now
            elif now - state.goku_absent_since >= 0.25:
                if not state.goku_armed:
                    state.goku_armed = True
            # Accumulate hold for return.
            if state.goku_armed and holds:
                if state.goku_hold_started is None:
                    state.goku_hold_started = now
                elif now - state.goku_hold_started >= GOKU_HOLD_SECONDS:
                    state.goku_hold_started = None
                    state.goku_armed = False
                    return "return"
        return None

    # Normal ready-phase accumulation.
    if not state.goku_armed:
        return None

    if holds:
        if state.goku_hold_started is None:
            state.goku_hold_started = now
        elif now - state.goku_hold_started >= GOKU_HOLD_SECONDS:
            state.goku_hold_started = None
            state.goku_armed = False
            return "vanish"
    else:
        # Reset hold if pose breaks (allow short dropouts < 0.3 s).
        if state.goku_hold_started is not None:
            if now - state.goku_hold_started < 0.3 or score > 0.0:
                pass  # Keep the start.
            else:
                state.goku_hold_started = None

    return None


# ---------------------------------------------------------------------------
# Goku effect — rendering
# ---------------------------------------------------------------------------

def make_goku_lines(
    height: int,
    width: int,
    count: int = 60,
    seed: int = 42,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Generate radial speed lines emanating from the image centre."""
    rng = np.random.default_rng(seed)
    cx, cy = width // 2, height // 2
    lines = []
    for _ in range(count):
        angle = rng.uniform(0, 2 * np.pi)
        r_inner = rng.uniform(0.08, 0.22) * min(height, width)
        r_outer = rng.uniform(0.45, 0.85) * min(height, width)
        x0 = int(cx + np.cos(angle) * r_inner)
        y0 = int(cy + np.sin(angle) * r_inner)
        x1 = int(cx + np.cos(angle) * r_outer)
        y1 = int(cy + np.sin(angle) * r_outer)
        lines.append(((x0, y0), (x1, y1)))
    return lines


def render_goku(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    age: float,
    effect_type: str = "vanish",
    lines: list | None = None,
) -> np.ndarray:
    """
    Render the Goku teleport effect.
    effect_type: 'vanish' fades out person, 'return' fades in.
    """
    height, width = frame.shape[:2]
    if lines is None:
        lines = make_goku_lines(height, width)

    progress = min(1.0, age / GOKU_EFFECT_SECONDS)

    if effect_type == "vanish":
        # Fade out: person dissolves, speed lines flash, then white flash.
        person_alpha = mask * max(0.0, 1.0 - progress * 1.8)
        person_alpha = np.clip(person_alpha, 0.0, 1.0)
        output = alpha_composite(frame, background, person_alpha)

        # Speed lines appear in the middle of the effect.
        line_strength = float(np.clip(progress * 3.0 - 0.3, 0.0, 1.0) *
                               np.clip(1.0 - (progress - 0.5) * 4.0, 0.0, 1.0))
        if line_strength > 0.01 and lines:
            line_layer = output.copy()
            for p0, p1 in lines:
                cv2.line(line_layer, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
            output = cv2.addWeighted(output, 1.0 - line_strength * 0.85,
                                     line_layer, line_strength * 0.85, 0)

        # White flash near the end.
        flash = float(np.clip((progress - 0.6) * 4.0, 0.0, 1.0))
        if flash > 0.0:
            white = np.full_like(output, 255)
            output = cv2.addWeighted(output, 1.0 - flash, white, flash, 0)

        return output

    else:  # return
        # Fade in: white flash first, then person materialises.
        flash = float(np.clip(1.0 - progress * 3.0, 0.0, 1.0))
        if flash > 0.0:
            white = np.full_like(background, 255)
            base = cv2.addWeighted(background, 1.0 - flash, white, flash, 0)
        else:
            base = background.copy()

        person_alpha = mask * min(1.0, max(0.0, progress * 2.0 - 0.5))
        output = alpha_composite(frame, base, person_alpha)

        # Speed lines in the beginning.
        line_strength = float(np.clip(progress * 3.0, 0.0, 1.0) *
                               np.clip(1.0 - progress * 2.5, 0.0, 1.0))
        if line_strength > 0.01 and lines:
            line_layer = output.copy()
            for p0, p1 in lines:
                cv2.line(line_layer, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
            output = cv2.addWeighted(output, 1.0 - line_strength * 0.85,
                                     line_layer, line_strength * 0.85, 0)
        return output


# ---------------------------------------------------------------------------
# Goku tracker (wraps hand + face landmarkers for the goku gesture)
# ---------------------------------------------------------------------------

class GokuTracker:
    """Detects the goku forehead-touch gesture using hand + face landmarks."""

    def __init__(self, hand_primary, hand_alternate, face_primary):
        self.hands = MultiViewHandLandmarker(hand_primary, hand_alternate)
        self.face = face_primary
        self._last_forehead: tuple[float, float, float, float] | None = None

    def recognize_goku(self, image: mp.Image, timestamp_ms: int) -> float:
        """Return a goku gesture score in [0, 1]."""
        face_result = self.face.detect_for_video(image, timestamp_ms)
        face_list = face_result.face_landmarks
        forehead = None
        if face_list:
            forehead = goku_forehead(face_list[0], image.width, image.height)
        if forehead is not None:
            self._last_forehead = forehead
        else:
            forehead = self._last_forehead

        hand_result = self.hands.detect_video(image, timestamp_ms)
        best_score = 0.0
        for hand_lm in hand_result.hand_landmarks:
            pts = np.array([(p.x * image.width, p.y * image.height, 0.0)
                            for p in hand_lm])
            s = goku_gesture_score(pts, forehead)
            if s > best_score:
                best_score = s
        return best_score

    def close(self) -> None:
        try:
            self.hands.close()
        finally:
            self.face.close()


# ---------------------------------------------------------------------------
# Model/task creation
# ---------------------------------------------------------------------------

def _make_hand_landmarker_options():
    if not HAND_MODEL.is_file():
        raise RuntimeError(f"Modelo das m\u00e3os n\u00e3o encontrado: {HAND_MODEL}")
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    )


def create_tasks(mode: str = "normal") -> tuple:
    """Create (segmenter, recognizer) for the given mode.

    Recognizer type:
      - "normal"  -> MultiViewGestureRecognizer
      - "naruto"  -> MultiViewHandLandmarker
      - "goku"    -> GokuTracker
      - "combo"   -> GokuTracker (handles both naruto+goku)
    """
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        if mode in ("naruto",):
            options = _make_hand_landmarker_options()
            primary = vision.HandLandmarker.create_from_options(options)
            try:
                alternate = vision.HandLandmarker.create_from_options(
                    replace(options, running_mode=vision.RunningMode.IMAGE)
                )
            except Exception:
                primary.close()
                raise
            recognizer = MultiViewHandLandmarker(primary, alternate)

        elif mode in ("goku", "combo"):
            hand_options = _make_hand_landmarker_options()
            h_primary = vision.HandLandmarker.create_from_options(hand_options)
            try:
                h_alternate = vision.HandLandmarker.create_from_options(
                    replace(hand_options, running_mode=vision.RunningMode.IMAGE)
                )
            except Exception:
                h_primary.close()
                raise
            if not FACE_MODEL.is_file():
                h_primary.close()
                h_alternate.close()
                raise RuntimeError(f"Modelo do rosto n\u00e3o encontrado: {FACE_MODEL}")
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.35,
                min_face_presence_confidence=0.35,
                min_tracking_confidence=0.45,
            )
            f_primary = vision.FaceLandmarker.create_from_options(face_options)
            recognizer = GokuTracker(h_primary, h_alternate, f_primary)

        else:  # normal
            classifier_options = mp.tasks.components.processors.ClassifierOptions(
                score_threshold=0.55,
                category_allowlist=["Victory"],
            )
            options = vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.5,
                canned_gesture_classifier_options=classifier_options,
            )
            primary = vision.GestureRecognizer.create_from_options(options)
            try:
                alternate = vision.GestureRecognizer.create_from_options(
                    replace(options, running_mode=vision.RunningMode.IMAGE)
                )
            except Exception:
                primary.close()
                raise
            recognizer = MultiViewGestureRecognizer(primary, alternate)

    except Exception:
        segmenter.close()
        raise

    return segmenter, recognizer


# ---------------------------------------------------------------------------
# SensorBundle — shared models for multi-effect sessions
# ---------------------------------------------------------------------------

@dataclass
class SensorBundle:
    segmenter: vision.ImageSegmenter
    hand_landmarker: MultiViewHandLandmarker | None = None
    gesture_recognizer: MultiViewGestureRecognizer | None = None
    face_landmarker: GokuTracker | None = None

    def close(self) -> None:
        errors = []
        for obj in (self.gesture_recognizer, self.hand_landmarker, self.face_landmarker):
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:
                    errors.append(exc)
        try:
            self.segmenter.close()
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise errors[0]


def create_sensor_bundle(active_effects: set[str]) -> SensorBundle:
    """Instantiate only the models needed for the selected effects."""
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        gesture_recognizer: MultiViewGestureRecognizer | None = None
        hand_landmarker: MultiViewHandLandmarker | None = None
        face_landmarker: GokuTracker | None = None

        if "normal" in active_effects:
            classifier_options = mp.tasks.components.processors.ClassifierOptions(
                score_threshold=0.55,
                category_allowlist=["Victory"],
            )
            g_options = vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.5,
                canned_gesture_classifier_options=classifier_options,
            )
            g_primary = vision.GestureRecognizer.create_from_options(g_options)
            g_alternate = vision.GestureRecognizer.create_from_options(
                replace(g_options, running_mode=vision.RunningMode.IMAGE)
            )
            gesture_recognizer = MultiViewGestureRecognizer(g_primary, g_alternate)

        needs_hands = "naruto" in active_effects or "goku" in active_effects
        if needs_hands:
            hand_options = _make_hand_landmarker_options()
            h_primary = vision.HandLandmarker.create_from_options(hand_options)
            h_alternate = vision.HandLandmarker.create_from_options(
                replace(hand_options, running_mode=vision.RunningMode.IMAGE)
            )

            if "goku" in active_effects:
                if not FACE_MODEL.is_file():
                    raise RuntimeError(f"Modelo do rosto n\u00e3o encontrado: {FACE_MODEL}")
                face_options = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.35,
                    min_face_presence_confidence=0.35,
                    min_tracking_confidence=0.45,
                )
                f_primary = vision.FaceLandmarker.create_from_options(face_options)
                face_landmarker = GokuTracker(h_primary, h_alternate, f_primary)
                # For naruto: reuse the same hand landmarker inside GokuTracker.
                if "naruto" in active_effects:
                    hand_landmarker = face_landmarker.hands
            else:
                hand_landmarker = MultiViewHandLandmarker(h_primary, h_alternate)

    except Exception:
        segmenter.close()
        raise

    return SensorBundle(
        segmenter=segmenter,
        hand_landmarker=hand_landmarker,
        gesture_recognizer=gesture_recognizer,
        face_landmarker=face_landmarker,
    )


# ---------------------------------------------------------------------------
# Fullscreen helper
# ---------------------------------------------------------------------------

def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> int:
    """Check every mode's models and audio decoding without opening a webcam."""
    required = (
        SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH,
        SAMPLE_TEST_IMAGE,
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Recursos ausentes: {', '.join(missing)}")
    load_log()

    # Decode both bundled and synthesized audio without requiring a physical
    # output device. This also catches missing pygame/MP3 support in the build.
    previous_audio_driver = os.environ.get("SDL_AUDIODRIVER")
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        with ExitStack() as resources:
            for sound_type in (VanishSound, TeleportSound):
                sound = sound_type()
                resources.callback(sound.close)
    finally:
        if previous_audio_driver is None:
            os.environ.pop("SDL_AUDIODRIVER", None)
        else:
            os.environ["SDL_AUDIODRIVER"] = previous_audio_driver

    for mode in ("normal", "naruto", "goku", "combo"):
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] Autodiagn\u00f3stico conclu\u00eddo: modelos, recursos e decodifica\u00e7\u00e3o de \u00e1udio verificados.")
    return 0


def test_image(image_path: Path, output_path: Path) -> int:
    """Run normal-mode segmentation on a still image and save a visual report."""
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None or not frame.size:
        raise RuntimeError(f"N\u00e3o foi poss\u00edvel abrir a imagem de teste: {image_path}")

    height, width = frame.shape[:2]
    download_models("normal")
    segmenter, recognizer = create_tasks("normal")
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mask = person_mask(segmenter, mp_image, 0, width, height)
        coverage = float(np.mean(mask >= 0.5))
        if coverage < 0.005 or coverage > 0.95:
            raise RuntimeError(
                "A segmenta\u00e7\u00e3o n\u00e3o separou a pessoa do fundo nesta imagem. "
                "Use uma imagem n\u00edtida, com uma pessoa vis\u00edvel e um pouco de fundo."
            )

        gesture, score = recognize_gesture(recognizer, mp_image, 0)
        background = np.full_like(frame, (52, 37, 86))
        extracted = alpha_composite(frame, background, mask)
        mask_view = cv2.cvtColor(
            np.clip(mask * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        views = [frame.copy(), extracted, mask_view]
        labels = ("ENTRADA", "RECORTE DA PESSOA", "MASCARA DE SEGMENTACAO")
        for view, label in zip(views, labels):
            cv2.putText(
                view, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )
        preview = np.concatenate(views, axis=1)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError(f"N\u00e3o foi poss\u00edvel salvar a pr\u00e9via: {output_path}")
    finally:
        try:
            recognizer.close()
        finally:
            segmenter.close()

    print(f"[Webcam] Segmenta\u00e7\u00e3o aprovada: \u00e1rea ocupada pela pessoa={coverage:.1%}.")
    gesture_label = {
        "Victory": "Sinal de V",
        "Hand_Detected": "Mao detectada",
        "No_Hand": "Nenhuma mao detectada",
    }.get(gesture, gesture)
    print(f"[Webcam] Resultado do reconhecimento de gesto nesta imagem: {gesture_label} ({score:.2f}).")
    print(f"[Webcam] Pr\u00e9via salva em: {output_path.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def _mode_to_effects(mode: str) -> set[str]:
    """Convert a legacy mode string to a set of effect names."""
    if mode == "combo":
        return {"naruto", "goku"}
    return {mode}


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    image_path = getattr(args, "test_image", None)
    if image_path is not None:
        return test_image(image_path, getattr(args, "test_output", DEFAULT_TEST_OUTPUT))

    # Determine active effects from --effects > --mode > interactive menu.
    active_effects: set[str] | None = None
    if args.effects is not None:
        try:
            active_effects = _parse_effects(args.effects)
        except argparse.ArgumentTypeError as exc:
            print(f"[VanishCam] ERRO: {exc}", file=sys.stderr)
            return 1
    elif args.mode is not None:
        active_effects = _mode_to_effects(args.mode)
    else:
        mode = choose_mode()
        if mode is None:
            return 0
        active_effects = _mode_to_effects(mode)

    with ExitStack() as resources:
        return run_camera(args, resources, active_effects)


def run_camera(args: argparse.Namespace, resources: ExitStack, active_effects: "str | set[str]") -> int:
    # Accept legacy string "mode" for backward compat (tests pass mode string).
    if isinstance(active_effects, str):
        active_effects = _mode_to_effects(active_effects)

    # For backward compat with tests that patch create_tasks and check mode.
    # Determine a single "mode" for create_tasks call.
    if active_effects == {"normal"}:
        mode = "normal"
    elif active_effects == {"naruto"}:
        mode = "naruto"
    elif active_effects == {"goku"}:
        mode = "goku"
    elif "naruto" in active_effects and "goku" in active_effects:
        mode = "combo"
    elif "naruto" in active_effects:
        mode = "naruto"
    elif "goku" in active_effects:
        mode = "goku"
    else:
        mode = "normal"

    download_models(active_effects)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)

    # Load naruto assets if needed.
    log_sprite = load_log() if "naruto" in active_effects else None
    vanish_sound: VanishSound | None = None
    teleport_sound: TeleportSound | None = None
    if "naruto" in active_effects:
        vanish_sound = VanishSound()
        resources.callback(vanish_sound.close)
    if "goku" in active_effects:
        teleport_sound = TeleportSound()
        resources.callback(teleport_sound.close)

    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(
        calibration_started=time.perf_counter(), fullscreen=args.fullscreen
    )
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    log_anchor: tuple[int, int, int] | None = None
    log_cache: dict[int, np.ndarray] = {}
    vanished_frame: np.ndarray | None = None
    goku_lines: list | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = 0.0
    started = time.perf_counter()

    # For cycling effects via 'V' key when multiple effects are active.
    _effects_cycle = sorted(active_effects)
    _v_cycle_index = 0

    print("[Webcam] Saia do enquadramento por 4 segundos para capturar o fundo.")
    print("[Webcam] Controles: ESPA\u00c7O captura o fundo, R reaparece, V ativa o efeito, F tela cheia, Q sai.")

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("A webcam parou de enviar imagens.")

            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            if "normal" in active_effects and (dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
            if "goku" in active_effects and goku_lines is None:
                goku_lines = make_goku_lines(height, width)
            now = time.perf_counter()

            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            # ----------------------------------------------------------------
            # Phase: calibrating
            # ----------------------------------------------------------------
            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Capturando o fundo: {remaining} segundos.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = "ready"
                    print("[Webcam] Fundo pronto. Voc\u00ea pode voltar ao enquadramento.")
                display = camera_frame

            elif background is None:
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            # ----------------------------------------------------------------
            # Phase: vanished (all effects)
            # ----------------------------------------------------------------
            elif state.phase == "vanished":
                if state.active_vanish == "goku":
                    # Goku: check for return gesture.
                    if now - state.last_gesture_check >= GOKU_INTERVAL_SECONDS:
                        state.last_gesture_check = now
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        score = recognizer.recognize_goku(mp_image, elapsed_ms)
                        action = update_goku_hold(state, score, now)
                        if action == "return":
                            # Start return animation.
                            state.phase = "goku_returning"
                            state.vanish_started = now
                            if teleport_sound is not None:
                                teleport_sound.play()
                    if vanished_frame is None:
                        vanished_frame = background.copy()
                    display = vanished_frame
                else:
                    # Normal / Naruto vanished.
                    if vanished_frame is None:
                        vanished_frame = background.copy()
                        if state.active_vanish == "naruto" and log_anchor is not None:
                            overlay_log(vanished_frame, log_sprite,
                                        landed_log_anchor(log_anchor, height),
                                        cache=log_cache)
                    display = vanished_frame

            # ----------------------------------------------------------------
            # Phase: goku_vanishing
            # ----------------------------------------------------------------
            elif state.phase == "goku_vanishing":
                age = max(0.0, time.perf_counter() - state.vanish_started)
                mask = last_mask
                if mask is None or now - last_mask_at >= 0.05:
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    last_mask = mask
                    last_mask_at = now
                display = render_goku(camera_frame, background, mask, age,
                                      effect_type="vanish", lines=goku_lines)
                if age >= GOKU_EFFECT_SECONDS:
                    state.phase = "goku_linger"
                    state.goku_linger_started = now
                    vanished_frame = background.copy()

            # ----------------------------------------------------------------
            # Phase: goku_linger (brief pause as white screen before vanished)
            # ----------------------------------------------------------------
            elif state.phase == "goku_linger":
                linger_age = now - state.goku_linger_started
                if linger_age >= GOKU_LINGER_SECONDS:
                    state.phase = "vanished"
                display = vanished_frame if vanished_frame is not None else background.copy()

            # ----------------------------------------------------------------
            # Phase: goku_returning
            # ----------------------------------------------------------------
            elif state.phase == "goku_returning":
                age = max(0.0, time.perf_counter() - state.vanish_started)
                mask = last_mask
                if mask is None or now - last_mask_at >= 0.05:
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    last_mask = mask
                    last_mask_at = now
                display = render_goku(camera_frame, background, mask, age,
                                      effect_type="return", lines=goku_lines)
                if age >= GOKU_EFFECT_SECONDS:
                    state.phase = "ready"
                    state.active_vanish = None
                    state.goku_armed = True
                    state.goku_hold_started = None
                    state.goku_absent_since = None
                    vanished_frame = None
                    last_mask = None
                    last_mask_at = float("-inf")

            # ----------------------------------------------------------------
            # All other phases (ready, vanishing, naruto_vanishing)
            # ----------------------------------------------------------------
            else:
                mask = last_mask

                # Gesture detection — only in ready phase.
                if state.phase == "ready":
                    check_interval = GOKU_INTERVAL_SECONDS if "goku" in active_effects else GESTURE_INTERVAL_SECONDS
                    if "naruto" in active_effects:
                        check_interval = 0.08
                    if now - state.last_gesture_check >= check_interval:
                        state.last_gesture_check = now
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                        if "normal" in active_effects:
                            name, score = recognize_gesture(recognizer, mp_image, elapsed_ms)
                            if args.gesture_debug and now - last_gesture_report >= 1.0:
                                gesture_label = {
                                    "Victory": "Sinal de V",
                                    "Hand_Detected": "M\u00e3o detectada",
                                    "No_Hand": "Nenhuma m\u00e3o detectada",
                                }.get(name, name)
                                print(
                                    f"[Gesto] {gesture_label}, pontua\u00e7\u00e3o={score:.2f}, "
                                    f"recorte={recognizer.preferred_crop or 'completo'}, "
                                    f"rota\u00e7\u00e3o={recognizer.preferred_rotation or 0}"
                                )
                                last_gesture_report = now
                            if start_vanish_on_gesture(state, name, score, now):
                                state.active_vanish = "normal"

                        if state.phase == "ready" and "naruto" in active_effects:
                            if mp_image is None:
                                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            # Get the hand landmarker — may be inside GokuTracker.
                            if "goku" in active_effects and isinstance(recognizer, GokuTracker):
                                hand_rec = recognizer.hands
                            elif isinstance(recognizer, MultiViewHandLandmarker):
                                hand_rec = recognizer
                            else:
                                hand_rec = None
                            if hand_rec is not None:
                                joined = hand_rec.recognize_joined_hands(mp_image, elapsed_ms)
                                if args.gesture_debug and now - last_gesture_report >= 1.0:
                                    print(f"[M\u00e3os] detectadas={hand_rec.last_hand_count}, "
                                          f"unidas={joined}, vista={hand_rec.preferred_view or 'completa'}")
                                    last_gesture_report = now
                                if start_naruto_on_join(state, joined, time.perf_counter()):
                                    state.active_vanish = "naruto"
                                    if vanish_sound is not None:
                                        vanish_sound.play()

                        if state.phase == "ready" and "goku" in active_effects:
                            if mp_image is None:
                                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            if isinstance(recognizer, GokuTracker):
                                goku_score = recognizer.recognize_goku(mp_image, elapsed_ms)
                            elif hasattr(recognizer, "recognize_goku"):
                                goku_score = recognizer.recognize_goku(mp_image, elapsed_ms)
                            else:
                                goku_score = 0.0
                            if args.gesture_debug and now - last_gesture_report >= 1.0:
                                print(f"[Goku] pontua\u00e7\u00e3o={goku_score:.2f}")
                                last_gesture_report = now
                            action = update_goku_hold(state, goku_score, now)
                            if action == "vanish":
                                state.phase = "goku_vanishing"
                                state.vanish_started = now
                                state.active_vanish = "goku"
                                if teleport_sound is not None:
                                    teleport_sound.play()

                # Mask update for active animations.
                if state.phase in ("vanishing", "naruto_vanishing"):
                    needs_mask = (
                        (state.active_vanish == "normal" and (mask is None or now - last_mask_at >= 0.05))
                        or (state.active_vanish == "naruto" and (log_anchor is None or now - state.vanish_started < 0.13))
                        or (mode == "normal" and (mask is None or now - last_mask_at >= 0.05))
                        or (mode == "naruto" and (log_anchor is None or now - state.vanish_started < 0.13))
                    )
                    if needs_mask:
                        if mp_image is None:
                            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        last_mask = mask
                        last_mask_at = now
                    active = state.active_vanish or mode
                    if active == "naruto" and log_anchor is None:
                        log_anchor = substitution_anchor(mask)

                if state.phase == "naruto_vanishing":
                    age = max(0.04, time.perf_counter() - state.vanish_started)
                    display = render_substitution(
                        camera_frame, background, mask, log_sprite, log_anchor, age, log_cache
                    )
                    if age >= EFFECT_SECONDS:
                        state.phase = "vanished"
                        vanished_frame = display
                elif state.phase == "ready":
                    display = camera_frame
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    if dissolve_field is not None and sparkle_noise is not None:
                        display = apply_dissolve(
                            camera_frame, background, mask, progress, dissolve_field, sparkle_noise
                        )
                    else:
                        display = camera_frame
                    if state.phase == "vanished":
                        vanished_frame = display

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = "calibrating"
                state.calibration_started = now
                state.vanish_started = None
                state.active_vanish = None
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] Saia do enquadramento por 4 segundos para capturar o fundo.")
            elif key == ord("r"):
                if background is not None:
                    state.phase = "ready"
                    state.vanish_started = None
                    state.active_vanish = None
                    state.goku_armed = True
                    state.goku_hold_started = None
                    state.goku_absent_since = None
                    vanished_frame = None
                    log_anchor = None
                    last_mask = None
                    last_mask_at = float("-inf")
            elif key == ord("v") and background is not None:
                # Cycle through effects if multiple are active.
                eff = _effects_cycle[_v_cycle_index % len(_effects_cycle)]
                _v_cycle_index += 1
                state.active_vanish = eff
                if eff == "naruto":
                    state.phase = "naruto_vanishing"
                    state.vanish_started = time.perf_counter()
                    if vanish_sound is not None:
                        vanish_sound.play()
                    log_anchor = None
                elif eff == "goku":
                    state.phase = "goku_vanishing"
                    state.vanish_started = time.perf_counter()
                    if teleport_sound is not None:
                        teleport_sound.play()
                else:  # normal
                    state.phase = "vanishing"
                    state.vanish_started = time.perf_counter()
                vanished_frame = None
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[VanishCam] ERRO: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
