"""Create the deterministic still-image fixture used by --test-image."""

from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 960, 540
OUTPUT = Path(__file__).with_name("sample_person.png")


def make_image() -> np.ndarray:
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    top = np.array((178, 204, 218), dtype=np.float32)[None, None, :]
    bottom = np.array((218, 225, 222), dtype=np.float32)[None, None, :]
    image = np.broadcast_to(top * (1 - y) + bottom * y, (HEIGHT, WIDTH, 3)).copy()

    # A simple indoor background gives the segmenter clear foreground edges.
    cv2.line(image, (0, 420), (WIDTH, 420), (157, 177, 180), 3, cv2.LINE_AA)
    for x in (90, 860):
        cv2.line(image, (x, 0), (x, 420), (188, 207, 215), 2, cv2.LINE_AA)
    cv2.ellipse(image, (482, 480), (135, 18), 0, 0, 360, (150, 162, 164), -1, cv2.LINE_AA)

    skin = (144, 180, 218)
    skin_light = (174, 202, 230)
    shirt = (82, 130, 77)
    shirt_light = (105, 155, 94)
    pants = (86, 79, 66)
    outline = (49, 55, 54)

    # Legs and shoes.
    cv2.fillPoly(image, [np.array(((405, 309), (489, 314), (478, 449), (444, 462), (420, 449)))], pants, cv2.LINE_AA)
    cv2.fillPoly(image, [np.array(((490, 314), (550, 306), (570, 449), (545, 463), (510, 448)))], pants, cv2.LINE_AA)
    cv2.line(image, (425, 440), (468, 440), outline, 3, cv2.LINE_AA)
    cv2.line(image, (519, 440), (561, 440), outline, 3, cv2.LINE_AA)
    cv2.ellipse(image, (439, 466), (42, 15), -5, 0, 360, (60, 68, 72), -1, cv2.LINE_AA)
    cv2.ellipse(image, (545, 466), (42, 15), 5, 0, 360, (60, 68, 72), -1, cv2.LINE_AA)

    # Torso and lowered arm.
    torso = np.array(((409, 177), (449, 160), (515, 160), (559, 186), (584, 321), (390, 321)))
    cv2.fillPoly(image, [torso], outline, cv2.LINE_AA)
    cv2.fillPoly(image, [np.array(((415, 183), (451, 169), (510, 169), (548, 191), (564, 310), (408, 310)))], shirt, cv2.LINE_AA)
    cv2.fillPoly(image, [np.array(((452, 176), (493, 176), (512, 306), (444, 306)))], shirt_light, cv2.LINE_AA)
    cv2.line(image, (417, 197), (366, 279), outline, 39, cv2.LINE_AA)
    cv2.line(image, (417, 197), (366, 279), skin, 29, cv2.LINE_AA)
    cv2.ellipse(image, (355, 291), (19, 25), 20, 0, 360, skin_light, -1, cv2.LINE_AA)

    # Raised arm and a large, clearly visible V hand.
    cv2.line(image, (548, 196), (617, 158), outline, 43, cv2.LINE_AA)
    cv2.line(image, (548, 196), (617, 158), skin, 33, cv2.LINE_AA)
    cv2.line(image, (615, 158), (666, 132), skin_light, 30, cv2.LINE_AA)
    cv2.ellipse(image, (679, 122), (29, 28), -25, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.line(image, (665, 110), (652, 47), skin, 14, cv2.LINE_AA)
    cv2.line(image, (687, 105), (700, 32), skin_light, 14, cv2.LINE_AA)
    cv2.line(image, (698, 119), (711, 133), skin_light, 10, cv2.LINE_AA)
    cv2.line(image, (660, 123), (641, 103), skin_light, 11, cv2.LINE_AA)

    # Neck, face, ears, and hair.
    cv2.rectangle(image, (458, 133), (505, 184), skin, -1)
    cv2.ellipse(image, (480, 104), (48, 62), 0, 0, 360, outline, -1, cv2.LINE_AA)
    cv2.ellipse(image, (481, 110), (39, 50), 0, 0, 360, skin_light, -1, cv2.LINE_AA)
    cv2.ellipse(image, (438, 112), (8, 13), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.ellipse(image, (524, 112), (8, 13), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.fillPoly(image, [np.array(((441, 84), (448, 57), (473, 43), (506, 48), (526, 70), (517, 86), (494, 74), (473, 82), (455, 75)))], outline, cv2.LINE_AA)
    cv2.line(image, (463, 105), (473, 105), outline, 3, cv2.LINE_AA)
    cv2.line(image, (491, 105), (501, 105), outline, 3, cv2.LINE_AA)
    cv2.line(image, (475, 133), (489, 133), (107, 125, 151), 2, cv2.LINE_AA)

    # Gentle, fixed texture keeps the sample reproducible and non-photographic.
    rng = np.random.default_rng(23)
    noise = rng.normal(0, 1.2, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    if not cv2.imwrite(str(OUTPUT), make_image()):
        raise SystemExit(f"Nao foi possivel salvar a imagem de exemplo: {OUTPUT}")
    print(OUTPUT)
