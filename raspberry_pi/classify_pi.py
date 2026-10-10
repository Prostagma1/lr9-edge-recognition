#!/usr/bin/env python3
"""Распознавание образов на Raspberry Pi — по исходникам из лекции 9.

Запускать НА Raspberry Pi (не на ПК). Делает две вещи:
  * бенчмарк: 250 классификаций одного изображения (как в лекции), печатает FPS;
  * (опционально) непрерывное распознавание кадров с камеры Pi.

Зависимости на Pi:
    pip install pillow "numpy<2"      # tflite-runtime собран под numpy 1.x
    pip install tflite-runtime        # лёгкий интерпретатор TFLite
    # для камеры (Pi OS Bookworm): sudo apt install -y python3-picamera2

Модель — квантованная MobileNetV2 под ImageNet (файл .tflite + labels.txt):
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
    tar xzf mobilenet_v2_1.0_224_quant.tgz
    # либо взять модель, скомпилированную под Edge TPU, с https://coral.ai/models/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:  # на ПК для проверки синтаксиса
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    load_delegate = None  # type: ignore

HERE = Path(__file__).resolve().parent


def make_interpreter(model_path: str, use_edgetpu: bool) -> Interpreter:
    if use_edgetpu and load_delegate is not None:
        # Coral Edge TPU: модель должна быть *_edgetpu.tflite
        return Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")],
        )
    return Interpreter(model_path=model_path, num_threads=4)


def load_labels(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def preprocess(image_path: str, size: int, quant: bool) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img)
    if quant:                       # uint8-модель ждёт сырые пиксели 0..255
        return arr[None, ...].astype(np.uint8)
    return ((arr / 127.5) - 1.0)[None, ...].astype(np.float32)  # [-1, 1]


def classify(interp: Interpreter, x: np.ndarray) -> int:
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(in_d["index"], x)
    interp.invoke()
    return int(np.argmax(interp.get_tensor(out_d["index"])[0]))


def benchmark(interp: Interpreter, x: np.ndarray, n: int, labels: list[str]) -> None:
    for _ in range(5):
        classify(interp, x)
    t0 = time.perf_counter()
    pred = 0
    for _ in range(n):
        pred = classify(interp, x)
    dt = time.perf_counter() - t0
    print(f"{n} классификаций за {dt:.2f} с  ->  {n / dt:.1f} FPS")
    if labels:
        print(f"предсказание: {labels[pred]}")


def camera_loop(interp: Interpreter, size: int, quant: bool, labels: list[str]) -> None:
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}))
    cam.start()
    print("Камера запущена. Ctrl+C для выхода.")
    try:
        while True:
            frame = cam.capture_array()
            img = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
            arr = np.asarray(img)
            x = (arr[None].astype(np.uint8) if quant
                 else ((arr / 127.5 - 1.0)[None].astype(np.float32)))
            t0 = time.perf_counter()
            pred = classify(interp, x)
            fps = 1.0 / (time.perf_counter() - t0)
            name = labels[pred] if labels else str(pred)
            print(f"\r{name:<30} {fps:5.1f} FPS", end="", flush=True)
    except KeyboardInterrupt:
        print("\nвыход")
    finally:
        cam.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(HERE / "mobilenet_v2_1.0_224_quant.tflite"))
    ap.add_argument("--labels", default=str(HERE / "labels.txt"))
    ap.add_argument("--image", default=str(HERE / "cat.jpg"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--runs", type=int, default=250)
    ap.add_argument("--float", action="store_true", help="модель во float (не quant)")
    ap.add_argument("--edgetpu", action="store_true", help="использовать Coral Edge TPU")
    ap.add_argument("--camera", action="store_true", help="непрерывно с камеры Pi")
    args = ap.parse_args()

    interp = make_interpreter(args.model, args.edgetpu)
    interp.allocate_tensors()
    labels = load_labels(args.labels) if Path(args.labels).exists() else []

    if args.camera:
        camera_loop(interp, args.size, not args.float, labels)
    else:
        x = preprocess(args.image, args.size, not args.float)
        benchmark(interp, x, args.runs, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
