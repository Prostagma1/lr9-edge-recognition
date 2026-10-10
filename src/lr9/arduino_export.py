"""Порт распознавания образов на Arduino Uno — вариант «цифра 8x8».

Uno не потянет MobileNetV2 / TFLite Micro (2 КБ SRAM). Берём самый маленький
осмысленный «образ» — рукописную цифру 8x8 (датасет sklearn `digits`, значения
0..16) — и распознаём int8-сетью 64->16->10 прямым проходом на C.

Камера НЕ нужна: изображение либо зашито в скетч (test_images.h), либо приходит
по Serial 64 байтами (см. `uv run lr9-send-digit`).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from lr9.tiny_mlp import (
    accuracy_int8,
    emit_model_h,
    emit_samples_h,
    forward_int8,
    quantize,
)

ARDUINO_DIR = Path(__file__).resolve().parents[2] / "arduino" / "digit_recognizer"
IN_DIM, HID_DIM, OUT_DIM = 64, 16, 10
X_SCALE = 1.0 / 16.0


def ascii_art(img: np.ndarray) -> list[str]:
    chars = " .:-=+*#%@"
    return ["// " + "".join(chars[min(9, int(v) * 9 // 16)] for v in row)
            for row in np.asarray(img).reshape(8, 8)]


def train_and_quantize():
    """-> (q, Xte, yte, float_acc). Используется и main(), и host-инструментом."""
    ds = load_digits()
    X = ds.images.reshape(len(ds.images), -1).astype(np.float32)   # 0..16
    y = ds.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(HID_DIM,), activation="relu", solver="adam",
                        alpha=1e-3, max_iter=800, random_state=0)
    clf.fit(Xtr * X_SCALE, ytr)
    float_acc = clf.score(Xte * X_SCALE, yte)
    q = quantize(clf.coefs_[0], clf.intercepts_[0], clf.coefs_[1], clf.intercepts_[1],
                 Xtr, x_scale=X_SCALE)
    return q, Xte, yte, float_acc


def main() -> int:
    q, Xte, yte, float_acc = train_and_quantize()
    print(f"[float] MLP 64-{HID_DIM}-10, test accuracy = {float_acc:.4f}")
    int8_acc = accuracy_int8(Xte, yte, q)
    print(f"[int8]  test accuracy = {int8_acc:.4f}  "
          f"(w1_scale={q['w1s']:.4g}, h_scale={q['h_scale']:.4g}, M1={q['M1']})")

    wbytes = emit_model_h(ARDUINO_DIR / "model_data.h", q, in_dim=IN_DIM, hid=HID_DIM,
                          out=OUT_DIM, float_acc=float_acc, int8_acc=int8_acc,
                          title="цифра 8x8 (sklearn digits)")

    # по одной картинке каждого класса из тестовой выборки
    imgs, labels, previews = [], [], []
    for cls in range(OUT_DIM):
        idx = np.where(yte == cls)[0]
        if len(idx):
            im = np.round(Xte[idx[0]]).clip(0, 16).astype(np.int8)
            imgs.append(im)
            labels.append(cls)
            previews += [f"// --- цифра {cls} ---", *ascii_art(im)]
    imgs = np.stack(imgs)
    labels = np.array(labels, np.uint8)
    emit_samples_h(ARDUINO_DIR / "test_images.h", imgs, labels, in_dim=IN_DIM,
                   previews=previews)

    print(f"\n[ok] {ARDUINO_DIR/'model_data.h'}  ({wbytes} B весов)")
    print(f"[ok] {ARDUINO_DIR/'test_images.h'}  ({len(labels)} изображений)")

    print("\n=== ожидаемый вывод скетча (тот же int8-алгоритм, что в .ino) ===")
    ok = 0
    for t, (im, lab) in enumerate(zip(imgs, labels)):
        pred = forward_int8(im.astype(np.int64), q)
        ok += pred == lab
        print(f"  img {t}: pred={pred} truth={lab} {'OK' if pred == lab else 'x'}")
    print(f"  итог: {ok}/{len(labels)}")
    print("\nДальше: прошить digit_recognizer.ino (arduino/README.md); "
          "без камеры — кормить цифры по Serial: uv run lr9-send-digit --port COM3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
