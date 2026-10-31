import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import mediapipe as mp

from vanish_cam import (
    AppState, GOKU_GESTURE_THRESHOLD, MultiViewGestureRecognizer, goku_forehead,
    goku_gesture_score, recognize_gesture, start_vanish_on_gesture, update_goku_hold,
    victory_landmark_score,
)
from vanish_sound import synthesize_teleport_wav


def hand_points(extended=(True, True, False, False)):
    points = np.zeros((21, 3))
    for base, x, opened, spread in zip(
        (5, 9, 13, 17), (-0.45, 0.0, 0.4, 0.75), extended, (-0.35, 0.25, 0.0, 0.0)
    ):
        if opened:
            points[base : base + 4] = [
                (x, 1.0, 0.0), (x + spread * 0.45, 1.7, 0.0),
                (x + spread * 0.75, 2.15, 0.0), (x + spread, 2.5, 0.0),
            ]
        else:
            points[base : base + 4] = [
                (x, 1.0, 0.0), (x, 1.5, 0.2),
                (x, 1.3, 0.5), (x, 0.95, 0.3),
            ]
    return points


def landmarks(points):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


class VictoryGestureTests(unittest.TestCase):
    def test_victory_with_back_palm_rotation_mirror_and_scale(self):
        points = hand_points()
        for angle in (0.0, 0.7, 1.57, 2.6, np.pi):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            tilt = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
            for mirror in (1, -1):
                for scale in (0.02, 1.0, 100.0):
                    with self.subTest(angle=angle, mirror=mirror, scale=scale):
                        transformed = points @ rotation @ tilt
                        transformed[:, 0] *= mirror
                        transformed = transformed * scale + (0.5, 0.3, 0.1)
                        self.assertGreater(victory_landmark_score(transformed), 0.62)

    def test_other_gestures_do_not_trigger(self):
        for fingers in (
            (False, False, False, False),  # Pugno.
            (True, True, True, True),  # Palmo aperto.
            (True, False, False, False),  # Solo indice.
            (True, False, False, True),  # Corna.
            (True, True, True, False),  # Tre dita.
        ):
            with self.subTest(fingers=fingers):
                self.assertEqual(victory_landmark_score(hand_points(fingers)), 0.0)

    def test_invalid_or_degenerate_landmarks(self):
        for points in (np.zeros((21, 3)), np.zeros((20, 3)), np.full((21, 3), np.nan)):
            self.assertEqual(victory_landmark_score(points), 0.0)

    def test_landmarks_trigger_without_classifier_category(self):
        result = SimpleNamespace(
            hand_world_landmarks=[landmarks(hand_points())], hand_landmarks=[], gestures=[]
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        name, score = recognize_gesture(recognizer, None, 1000)
        self.assertEqual(name, "Victory")
        self.assertGreater(score, 0.62)

    def test_normalized_landmark_fallback_handles_video_aspect_ratio(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        normalized = points / (image.width, image.height, image.width)
        result = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[landmarks(normalized)], gestures=[]
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_existing_classifier_still_works(self):
        result = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]],
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, None, 1000), ("Victory", 0.9))

    def test_occluded_folded_fingers_can_appear_short_and_straight(self):
        points = hand_points()
        for base in (13, 17):
            points[base : base + 4] = [
                (0.4, 1.0, 0.0), (0.4, 1.1, 0.0),
                (0.4, 1.2, 0.0), (0.4, 1.3, 0.0),
            ]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_hidden_fingers_with_coincident_landmarks(self):
        points = hand_points()
        points[13:16] = points[13]
        points[16] = points[13]
        points[17:20] = points[17]
        points[20] = points[17]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_visible_geometry_is_used_even_with_inaccurate_world_landmarks(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        world = hand_points()
        world[8] = world[6]
        self.assertEqual(victory_landmark_score(world), 0.0)
        result = SimpleNamespace(
            hand_world_landmarks=[landmarks(world)],
            hand_landmarks=[landmarks(points / (960, 540, 960))], gestures=[],
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_diagnostics_distinguish_missing_hand_and_rejected_gesture(self):
        for hands, expected in (([], "No_Hand"), ([landmarks(hand_points((False,) * 4))], "Hand_Detected")):
            result = SimpleNamespace(hand_world_landmarks=hands, hand_landmarks=[], gestures=[])
            recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
            self.assertEqual(recognize_gesture(recognizer, None, 1000)[0], expected)


class ImmediateAnimationTests(unittest.TestCase):
    def test_first_recognized_v_starts_animation_immediately(self):
        state = AppState(phase="ready")
        self.assertTrue(start_vanish_on_gesture(state, "Victory", 0.9, 12.5))
        self.assertEqual(state.phase, "vanishing")
        self.assertEqual(state.vanish_started, 12.5)

    def test_other_or_weak_gestures_do_not_vanish(self):
        for name, score in (("No_Hand", 0.0), ("Hand_Detected", 0.0), ("Victory", 0.61)):
            with self.subTest(name=name, score=score):
                state = AppState(phase="ready")
                self.assertFalse(start_vanish_on_gesture(state, name, score, 12.5))
                self.assertEqual(state.phase, "ready")
                self.assertIsNone(state.vanish_started)

    def test_only_ready_phase_responds(self):
        for phase in ("calibrating", "vanishing", "vanished"):
            state = AppState(phase=phase)
            self.assertFalse(start_vanish_on_gesture(state, "Victory", 0.9, 12.5))
            self.assertEqual(state.phase, phase)


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.empty = SimpleNamespace(hand_world_landmarks=[], hand_landmarks=[], gestures=[])
        self.victory = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]],
        )
        self.image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.zeros((32, 48, 3), dtype=np.uint8)
        )

    def test_rotated_detection_and_preferred_view_are_used(self):
        calls = []

        def recognize(image):
            calls.append((image.width, image.height))
            return self.victory if image.width == 32 else self.empty

        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty),
            SimpleNamespace(recognize=recognize),
        )
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [(48, 32), (32, 48)])
        self.assertEqual(detector.preferred_rotation, 1)
        calls.clear()
        self.assertEqual(recognize_gesture(detector, self.image, 1140)[0], "Victory")
        self.assertEqual(calls, [(32, 48)])

    def test_failed_search_is_throttled_without_reusing_old_positive(self):
        calls = []
        alternate = SimpleNamespace(recognize=lambda image: calls.append(image) or self.empty)
        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty), alternate
        )
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "No_Hand")
        self.assertEqual(len(calls), 3)
        recognize_gesture(detector, self.image, 1070)  # Dentro da janela de espera.
        self.assertEqual(len(calls), 3)
        alternate.recognize = lambda image: self.victory
        self.assertEqual(recognize_gesture(detector, self.image, 1140)[0], "Victory")
        alternate.recognize = lambda image: self.empty
        self.assertEqual(recognize_gesture(detector, self.image, 1560)[0], "No_Hand")

    def test_primary_victory_avoids_extra_inference(self):
        calls = []
        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.victory),
            SimpleNamespace(recognize=lambda image: calls.append(image) or self.empty),
        )
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [])

    def test_lower_hand_crop_recovers_a_small_sideways_v(self):
        calls = []

        def recognize(image):
            calls.append((image.width, image.height))
            return self.victory if (image.width, image.height) == (37, 24) else self.empty

        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty),
            SimpleNamespace(recognize=recognize),
        )
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "No_Hand")
        self.assertIsNone(detector.preferred_crop)
        self.assertEqual(recognize_gesture(detector, self.image, 1140)[0], "Victory")
        self.assertEqual(detector.preferred_crop, "lower_left")
        self.assertEqual(detector.preferred_rotation, 0)
        calls.clear()
        self.assertEqual(recognize_gesture(detector, self.image, 1280)[0], "Victory")
        self.assertEqual(calls, [(37, 24)])


def goku_points(
    scale=60.0, origin=(320.0, 180.0), spread=0.08,
    ring_extended=False, pinky_extended=False,
):
    points = np.zeros((21, 3))
    points[0] = (0.0, 1.0, 0.0)  # Polso.
    for base, x, opened in (
        (5, -spread, True), (9, spread, True),
        (13, 0.35, ring_extended), (17, 0.6, pinky_extended),
    ):
        if opened:
            points[base:base + 4] = [
                (x, 0.0, 0.0), (x, -0.6, 0.0), (x, -1.1, 0.0), (x, -1.5, 0.0),
            ]
        else:
            points[base:base + 4] = [
                (x, 0.0, 0.0), (x, 0.45, 0.2), (x, 0.2, 0.5), (x, -0.1, 0.3),
            ]
    return points * scale + (origin[0], origin[1], 0.0)


def goku_forehead_for(scale=60.0, origin=(320.0, 180.0), face_height=300.0):
    return (origin[0], origin[1] - 1.5 * scale, face_height, face_height * 0.8)


class GokuGestureTests(unittest.TestCase):
    def test_two_fingers_together_on_the_forehead_score_above_threshold(self):
        score = goku_gesture_score(goku_points(), goku_forehead_for())
        self.assertGreater(score, GOKU_GESTURE_THRESHOLD)

    def test_open_v_and_other_fingers_do_not_trigger(self):
        forehead = goku_forehead_for()
        for points in (
            goku_points(spread=0.6),  # V aberta do modo normal.
            goku_points(ring_extended=True),
            goku_points(pinky_extended=True),
            np.zeros((21, 3)),
            np.zeros((20, 3)),
            np.full((21, 3), np.nan),
        ):
            with self.subTest(shape=points.shape, spread=points[5, 0]):
                self.assertEqual(goku_gesture_score(points, forehead), 0.0)

    def test_fingers_far_from_the_forehead_do_not_trigger(self):
        points = goku_points(origin=(320.0, 500.0))
        self.assertEqual(goku_gesture_score(points, goku_forehead_for()), 0.0)

    def test_forehead_anchor_uses_face_landmarks_and_rejects_tiny_faces(self):
        face = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(468)]
        face[10] = SimpleNamespace(x=0.5, y=0.2, z=0.0)
        face[152] = SimpleNamespace(x=0.5, y=0.6, z=0.0)
        face[234] = SimpleNamespace(x=0.3, y=0.4, z=0.0)
        face[454] = SimpleNamespace(x=0.7, y=0.4, z=0.0)
        x, y, height, width = goku_forehead(face, 200, 100)
        self.assertAlmostEqual(x, 100.0)
        self.assertAlmostEqual(y, 20.0)
        self.assertAlmostEqual(height, 40.0)
        self.assertAlmostEqual(width, 80.0)
        face[152] = SimpleNamespace(x=0.5, y=0.26, z=0.0)
        self.assertIsNone(goku_forehead(face, 200, 100))
        self.assertIsNone(goku_forehead(face[:20], 200, 100))


class GokuHoldTests(unittest.TestCase):
    def test_one_second_hold_vanishes_and_absence_rearms_for_return(self):
        state = AppState(phase="ready")
        self.assertIsNone(update_goku_hold(state, 0.9, 10.0))
        self.assertIsNone(update_goku_hold(state, 0.9, 10.6))
        self.assertEqual(update_goku_hold(state, 0.9, 11.0), "vanish")
        self.assertFalse(state.goku_armed)

        # A mão continua na testa: não pode voltar antes de ser retirada.
        state.phase = "vanished"
        self.assertIsNone(update_goku_hold(state, 0.9, 11.5))
        self.assertIsNone(update_goku_hold(state, 0.0, 12.0))
        self.assertIsNone(update_goku_hold(state, 0.0, 12.2))
        self.assertFalse(state.goku_armed)
        self.assertIsNone(update_goku_hold(state, 0.0, 12.3))
        self.assertTrue(state.goku_armed)

        self.assertIsNone(update_goku_hold(state, 0.9, 12.4))
        self.assertEqual(update_goku_hold(state, 0.9, 13.4), "return")
        self.assertFalse(state.goku_armed)

    def test_weak_or_interrupted_hold_does_not_trigger(self):
        state = AppState(phase="ready")
        self.assertIsNone(update_goku_hold(state, GOKU_GESTURE_THRESHOLD - 0.01, 10.0))
        self.assertIsNone(state.goku_hold_started)
        self.assertIsNone(update_goku_hold(state, 0.9, 10.0))
        self.assertIsNone(update_goku_hold(state, 0.0, 10.5))
        self.assertIsNone(state.goku_hold_started)
        self.assertIsNone(update_goku_hold(state, 0.9, 10.6))
        self.assertIsNone(update_goku_hold(state, 0.9, 11.5))
        self.assertEqual(update_goku_hold(state, 0.9, 11.6), "vanish")

    def test_brief_tracking_dropout_does_not_restart_the_hold(self):
        state = AppState(phase="ready")
        self.assertIsNone(update_goku_hold(state, 0.9, 10.0))
        self.assertIsNone(update_goku_hold(state, 0.0, 10.2))  # Quadro perdido.
        self.assertIsNone(update_goku_hold(state, 0.9, 10.4))
        self.assertEqual(update_goku_hold(state, 0.9, 11.0), "vanish")


class TeleportSoundTests(unittest.TestCase):
    def test_synthesized_fallback_is_a_valid_wav(self):
        with TemporaryDirectory() as folder:
            path = synthesize_teleport_wav(Path(folder) / "teleport.wav", duration=0.25)
            self.assertTrue(path.is_file())
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertEqual(audio.getframerate(), 44100)
                frames = audio.readframes(audio.getnframes())
        self.assertGreater(len(frames), 1000)
        self.assertNotEqual(set(frames), {0})


if __name__ == "__main__":
    unittest.main()
