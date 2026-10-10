"""Host-инструменты для порта на Arduino Uno — кормят плату «образом» по Serial.

Камера/датчики не нужны: изображение цифры (64 байта 0..16) или окно сигнала
(32 байта 0..16) генерируется на ПК и отправляется в COM-порт; плата отвечает
распознанным классом.

Без указанного `--port` работает как «сухой прогон»: считает предсказание тем же
int8-алгоритмом, что зашит в скетч, и показывает, что должна была бы ответить плата.

    uv run lr9-send-digit  --port COM3            # случайная цифра из sklearn digits
    uv run lr9-send-digit  --index 42             # конкретный пример
    uv run lr9-send-digit  --sweep                # все 10 тестовых цифр
    uv run lr9-send-signal --port COM3 --shape triangle
    uv run lr9-send-signal --shape random --sweep
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from lr9.tiny_mlp import forward_int8


# --------------------------------------------------------------------- serial
def _talk(port: str, baud: int, payload: bytes, read_secs: float = 1.5) -> str:
    import serial  # pyserial

    with serial.Serial(port, baud, timeout=0.3) as ser:
        time.sleep(2.0)                 # автосброс платы при открытии порта
        ser.reset_input_buffer()
        ser.write(payload)
        ser.flush()
        deadline = time.time() + read_secs
        out = []
        while time.time() < deadline:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                out.append(line)
        return "\n".join(out)


def _ascii_img(v: np.ndarray) -> str:
    chars = " .:-=+*#%@"
    return "\n".join("".join(chars[min(9, int(p) * 9 // 16)] for p in row)
                     for row in v.reshape(8, 8))


def _ascii_sig(v: np.ndarray) -> str:
    rows = []
    for lvl in range(16, -1, -4):
        rows.append("".join("#" if p >= lvl else " " for p in v))
    return "\n".join(rows)


# --------------------------------------------------------------------- digit
def send_digit() -> int:
    from lr9.arduino_export import train_and_quantize

    ap = argparse.ArgumentParser(description="Отправить цифру 8x8 на Arduino Uno")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--index", type=int, help="индекс примера из sklearn digits")
    ap.add_argument("--sweep", action="store_true", help="прогнать 10 тестовых цифр")
    args = ap.parse_args()

    q, Xte, yte, _ = train_and_quantize()

    if args.sweep:
        picks = [np.where(yte == c)[0][0] for c in range(10)]
    elif args.index is not None:
        picks = [args.index]
    else:
        picks = [int(np.random.default_rng().integers(len(Xte)))]

    for idx in picks:
        img = np.round(Xte[idx]).clip(0, 16).astype(np.int64)
        truth = int(yte[idx])
        print(f"\n--- пример {idx}, истинная цифра {truth} ---")
        print(_ascii_img(img))
        expected = forward_int8(img, q)
        print(f"ожидается (int8-алгоритм скетча): {expected}")
        if args.port:
            reply = _talk(args.port, args.baud, bytes(int(v) for v in img))
            print(f"ответ платы:\n{reply}")
        else:
            print("(--port не задан — сухой прогон; для реальной платы: --port COM3)")
    return 0


# -------------------------------------------------------------------- signal
def send_signal() -> int:
    from lr9.signal_export import CLASSES, _make

    ap = argparse.ArgumentParser(description="Отправить окно сигнала на Arduino Uno")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--shape", default="random",
                    choices=[*CLASSES, "random"])
    ap.add_argument("--sweep", action="store_true", help="по одному окну каждой формы")
    args = ap.parse_args()

    from lr9.signal_export import train_and_quantize

    q, _, _, _ = train_and_quantize()

    shapes = list(range(len(CLASSES))) if args.sweep else [
        int(np.random.default_rng().integers(len(CLASSES))) if args.shape == "random"
        else CLASSES.index(args.shape)]

    for c in shapes:
        win = np.round(_make(c)).clip(0, 16).astype(np.int64)
        print(f"\n--- форма: {CLASSES[c]} ---")
        print(_ascii_sig(win))
        expected = forward_int8(win, q)
        print(f"ожидается: {CLASSES[expected]}")
        if args.port:
            reply = _talk(args.port, args.baud, bytes(int(v) for v in win))
            print(f"ответ платы:\n{reply}")
        else:
            print("(--port не задан — сухой прогон)")
    return 0


if __name__ == "__main__":
    raise SystemExit(send_digit())
