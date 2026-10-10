"""Порт на Arduino Uno — вариант «форма сигнала на A0» (совсем без датчиков).

«Образ» здесь — форма аналогового сигнала: 32 отсчёта с входа A0, нормированные в
0..16. Классы: sine, square, triangle, sawtooth, flat, noise. Модель — int8-MLP
32->24->6, обучается на синтетических волнах, работает прямым проходом на C.

Что подать на A0:
  * потенциометр — рукой рисуете ramp/triangle/flat;
  * функциональный генератор / линейный выход звука через делитель;
  * PWM-пин через RC-фильтр (сам Uno) — можно сгенерировать sine/square;
  * ничего (плавающий вход) — получится noise / 50 Гц наводка.
Либо вообще без железа — прислать 32 байта по Serial: `uv run lr9-send-signal`.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from lr9.tiny_mlp import (
    accuracy_int8,
    emit_model_h,
    emit_samples_h,
    forward_int8,
    quantize,
)

ARDUINO_DIR = Path(__file__).resolve().parents[2] / "arduino" / "signal_shape"
WIN, HID, N_CLASS = 32, 24, 6
CLASSES = ["sine", "square", "triangle", "sawtooth", "flat", "noise"]
X_SCALE = 1.0 / 16.0
RNG = np.random.default_rng(0)


def _norm(w: np.ndarray) -> np.ndarray:
    """min-max в 0..16 — ровно так же нормирует скетч после чтения A0."""
    lo, hi = w.min(), w.max()
    if hi - lo < 1e-6:
        return np.full_like(w, 8.0)
    return (w - lo) / (hi - lo) * 16.0


def _make(cls: int) -> np.ndarray:
    t = np.linspace(0, 1, WIN, endpoint=False)
    cycles = RNG.uniform(1.0, 3.5)
    phase = RNG.uniform(0, 2 * np.pi)
    if cls == 0:      # sine
        w = np.sin(2 * np.pi * cycles * t + phase)
    elif cls == 1:    # square
        w = np.sign(np.sin(2 * np.pi * cycles * t + phase))
    elif cls == 2:    # triangle
        w = 2 * np.abs(2 * ((cycles * t + phase / (2 * np.pi)) % 1) - 1) - 1
    elif cls == 3:    # sawtooth
        w = 2 * ((cycles * t + phase / (2 * np.pi)) % 1) - 1
    elif cls == 4:    # flat
        w = np.zeros(WIN) + RNG.uniform(-0.15, 0.15)
    else:             # noise
        w = RNG.uniform(-1, 1, WIN)
    w = w + RNG.normal(0, 0.03, WIN)          # шум измерения
    return _norm(w)


def build_dataset(n_per_class: int = 3000):
    X = np.stack([_make(c) for c in range(N_CLASS) for _ in range(n_per_class)])
    y = np.repeat(np.arange(N_CLASS), n_per_class)
    return X.astype(np.float32), y.astype(np.int64)


def ascii_row(w: np.ndarray) -> list[str]:
    chars = "_.-=*#"
    return ["// " + "".join(chars[min(5, int(v) * 5 // 16)] for v in w)]


def train_and_quantize():
    X, y = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(HID,), activation="relu", solver="adam",
                        alpha=1e-3, max_iter=900, random_state=0)
    clf.fit(Xtr * X_SCALE, ytr)
    float_acc = clf.score(Xte * X_SCALE, yte)
    q = quantize(clf.coefs_[0], clf.intercepts_[0], clf.coefs_[1], clf.intercepts_[1],
                 Xtr, x_scale=X_SCALE)
    return q, Xte, yte, float_acc


def main() -> int:
    q, Xte, yte, float_acc = train_and_quantize()
    print(f"[float] MLP {WIN}-{HID}-{N_CLASS}, test accuracy = {float_acc:.4f}")
    int8_acc = accuracy_int8(Xte, yte, q)
    print(f"[int8]  test accuracy = {int8_acc:.4f}  (M1={q['M1']})")

    wbytes = emit_model_h(ARDUINO_DIR / "signal_model.h", q, in_dim=WIN, hid=HID,
                          out=N_CLASS, float_acc=float_acc, int8_acc=int8_acc,
                          title="форма сигнала на A0 (32 отсчёта)")

    # генерируем строку имён классов для скетча (отдельным include)
    (ARDUINO_DIR / "signal_labels.h").write_text(
        "#pragma once\n#include <avr/pgmspace.h>\n"
        "const char* const SHAPE_NAMES[OUT_DIM] = { "
        + ", ".join(f'"{c}"' for c in CLASSES) + " };\n", encoding="utf-8")

    # по одному тестовому окну каждого класса
    wins, labs, prev = [], [], []
    for c in range(N_CLASS):
        idx = np.where(yte == c)[0][0]
        w = np.round(Xte[idx]).clip(0, 16).astype(np.int8)
        wins.append(w); labs.append(c)
        prev += [f"// --- {CLASSES[c]} ---", *ascii_row(w)]
    wins = np.stack(wins); labs = np.array(labs, np.uint8)
    emit_samples_h(ARDUINO_DIR / "test_signals.h", wins, labs, in_dim=WIN, previews=prev,
                   images_name="TEST_SIGNALS", labels_name="TEST_LABELS")

    print(f"\n[ok] {ARDUINO_DIR/'signal_model.h'}  ({wbytes} B весов)")
    print(f"[ok] {ARDUINO_DIR/'signal_labels.h'}, {ARDUINO_DIR/'test_signals.h'}")

    print("\n=== ожидаемый вывод скетча ===")
    ok = 0
    for i, (w, lab) in enumerate(zip(wins, labs)):
        pred = forward_int8(w.astype(np.int64), q)
        ok += pred == lab
        print(f"  win {i}: pred={CLASSES[pred]:<9} truth={CLASSES[lab]:<9} "
              f"{'OK' if pred == lab else 'x'}")
    print(f"  итог: {ok}/{len(labs)}")
    print("\nДальше: прошить signal_shape.ino; без железа — прислать окно по Serial: "
          "uv run lr9-send-signal --port COM3 --shape triangle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
