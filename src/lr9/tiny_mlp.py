"""Общий код для крошечных int8-MLP, запускаемых прямым проходом на Arduino Uno.

Схема квантования (та же, что в TFLite):
  * вход и веса — int8, аккумулятор — int32;
  * между слоями ReLU + requant: h = clip((acc * M + 2^(SHIFT-1)) >> SHIFT, 0, 127),
    где M — фиксированный множитель (x_scale * w1_scale / h_scale);
  * на выходе — argmax по int32 (requant не нужен, масштаб монотонный).

Python-функция `forward_int8` — побитовый эквивалент C-кода в .ino, поэтому её
точность = ожидаемому результату на плате.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SHIFT = 24


def quantize(W1, B1, W2, B2, x_train, *, x_scale: float) -> dict:
    """W1: (in, hid), W2: (hid, out). x_train — сырые входы в том же масштабе, что x_scale."""
    W1 = np.asarray(W1, np.float64)
    W2 = np.asarray(W2, np.float64)
    B1 = np.asarray(B1, np.float64)
    B2 = np.asarray(B2, np.float64)

    def qmat(m):
        s = np.abs(m).max() / 127.0
        return np.clip(np.round(m / s), -127, 127).astype(np.int8), s

    W1q, w1s = qmat(W1)
    W2q, w2s = qmat(W2)

    h_float = np.maximum(0.0, (x_train * x_scale) @ W1 + B1)
    h_scale = float(h_float.max()) / 127.0
    B1q = np.round(B1 / (x_scale * w1s)).astype(np.int32)
    B2q = np.round(B2 / (h_scale * w2s)).astype(np.int32)
    M1 = int(round((x_scale * w1s) / h_scale * (1 << SHIFT)))

    return {"W1q": W1q, "B1q": B1q, "M1": M1, "W2q": W2q, "B2q": B2q,
            "w1s": w1s, "w2s": w2s, "h_scale": h_scale}


def forward_int8(x_i8: np.ndarray, q: dict) -> int:
    """x_i8: (in,) целые значения того же диапазона, что подаёт скетч. -> класс."""
    x = x_i8.astype(np.int64)
    acc1 = q["B1q"].astype(np.int64) + x @ q["W1q"].astype(np.int64)   # (hid,)
    acc1 = np.maximum(acc1, 0)
    h = np.clip((acc1 * q["M1"] + (1 << (SHIFT - 1))) >> SHIFT, 0, 127)
    acc2 = q["B2q"].astype(np.int64) + h @ q["W2q"].astype(np.int64)   # (out,)
    return int(np.argmax(acc2))


def accuracy_int8(X: np.ndarray, y: np.ndarray, q: dict) -> float:
    preds = np.array([forward_int8(np.round(x).astype(np.int64), q) for x in X])
    return float(np.mean(preds == y))


def _c_array(name: str, arr: np.ndarray, ctype: str, dims: str | None = None) -> str:
    flat = arr.flatten()
    body = ", ".join(str(int(v)) for v in flat)
    idx = dims if dims else f"[{flat.size}]"
    return f"const {ctype} {name}{idx} PROGMEM = {{ {body} }};"


def emit_model_h(path: Path, q: dict, *, in_dim: int, hid: int, out: int,
                 float_acc: float, int8_acc: float, title: str) -> int:
    total = q["W1q"].size + q["B1q"].size + q["W2q"].size + q["B2q"].size
    wbytes = q["W1q"].nbytes + q["W2q"].nbytes + q["B1q"].nbytes + q["B2q"].nbytes
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"// Автогенерировано lr9. {title}",
        f"// int8 MLP {in_dim}-{hid}-{out}. float acc={float_acc:.4f}, int8 acc={int8_acc:.4f}.",
        f"// параметров: {total}, вес в PROGMEM: {wbytes} байт.",
        "#pragma once",
        "#include <Arduino.h>",
        "#include <avr/pgmspace.h>",
        "",
        f"#define IN_DIM  {in_dim}",
        f"#define HID_DIM {hid}",
        f"#define OUT_DIM {out}",
        f"#define REQUANT_SHIFT {SHIFT}",
        f"const int32_t REQUANT_M1 = {q['M1']};",
        "",
        _c_array("W1", q["W1q"].T, "int8_t"),   # (hid, in) — построчно по нейрону
        _c_array("B1", q["B1q"], "int32_t"),
        _c_array("W2", q["W2q"].T, "int8_t"),   # (out, hid)
        _c_array("B2", q["B2q"], "int32_t"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return wbytes


def emit_samples_h(path: Path, samples: np.ndarray, labels: np.ndarray, *,
                   in_dim: int, previews: list[str] | None = None,
                   images_name: str = "TEST_IMAGES", labels_name: str = "TEST_LABELS") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ",\n  ".join("{ " + ", ".join(str(int(v)) for v in s) + " }" for s in samples)
    lines = [
        "// Автогенерировано lr9. Подключать ПОСЛЕ *_model.h (нужны IN_DIM и типы).",
        "#pragma once", "#include <Arduino.h>", "#include <avr/pgmspace.h>", "",
        f"#define N_TEST {len(labels)}", "",
    ]
    if previews:
        lines += previews + [""]
    lines += [
        f"const int8_t {images_name}[N_TEST][IN_DIM] PROGMEM = {{\n  {rows}\n}};",
        f"const uint8_t {labels_name}[N_TEST] PROGMEM = {{ "
        + ", ".join(str(int(v)) for v in labels) + " };",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
