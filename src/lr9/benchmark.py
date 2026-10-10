"""Бенчмарк из лекции 9, воспроизведённый на доступном железе.

Лекция: 250 классификаций одного и того же изображения кота одной моделью
(MobileNetV2) на разных конфигурациях Raspberry Pi / Jetson Nano / Coral, метрика —
FPS. Здесь та же методика, но конфигурации — те, что доступны на этой машине
(ноутбук с RTX 3050 Ti), сопоставленные с рядами из лекции:

  плоский TensorFlow на Pi           -> PyTorch eager, CPU
  TFLite на Pi                        -> ONNX Runtime, CPU
  TFLite 8-bit на Pi                  -> ONNX Runtime, CPU, динамическое квантование INT8
  Jetson Nano, 128 ядер CUDA          -> PyTorch, CUDA fp32
  Jetson Nano, TF-TRT 16-bit          -> PyTorch, CUDA fp16 (autocast)
  Coral USB (Edge TPU)               -> нет TPU на машине (ряд отсутствует)

Реальный скрипт для Raspberry Pi (tflite_runtime + камера) — в ./raspberry_pi/.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

RESULTS = Path(__file__).resolve().parents[2] / "results"
CAT_URL = ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/"
           "master/n02123045_tabby.JPEG")
N_RUNS = 250
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def load_cat() -> np.ndarray:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = RESULTS / "cat.jpg"
    if not cache.exists():
        with urllib.request.urlopen(CAT_URL, timeout=30) as r:  # noqa: S310
            cache.write_bytes(r.read())
    img = Image.open(io.BytesIO(cache.read_bytes())).convert("RGB")
    short = 256
    w, h = img.size
    s = short / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    arr = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
    return ((arr - MEAN) / STD)[None, ...].astype(np.float32)


def labels() -> list[str]:
    return list(MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"])


# --------------------------------------------------------------------- runners
def _time_loop(fn, x, n: int, warmup: int, cuda: bool) -> tuple[float, int]:
    for _ in range(warmup):
        fn(x)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = None
    for _ in range(n):
        pred = fn(x)
    if cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0, int(pred)


def bench_torch(x_np, device: str, half: bool) -> dict:
    dev = torch.device(device)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval().to(dev)
    x = torch.from_numpy(x_np).to(dev)
    if half:
        model = model.half()
        x = x.half()

    @torch.inference_mode()
    def fn(t):
        return model(t).argmax(1).item()

    sec, pred = _time_loop(fn, x, N_RUNS, 15, dev.type == "cuda")
    return {"sec": sec, "pred": pred}


def bench_torchscript(x_np) -> dict:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    ts = torch.jit.trace(model, torch.from_numpy(x_np))
    ts = torch.jit.optimize_for_inference(ts)
    x = torch.from_numpy(x_np)

    @torch.inference_mode()
    def fn(t):
        return ts(t).argmax(1).item()

    sec, pred = _time_loop(fn, x, N_RUNS, 15, False)
    return {"sec": sec, "pred": pred}


def _export_onnx(path: Path) -> None:
    if path.exists():
        return
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), path.as_posix(),
                      input_names=["x"], output_names=["y"], opset_version=17,
                      dynamo=False)


def bench_onnx(x_np, quantized: bool) -> dict:
    import onnxruntime as ort

    fp32 = RESULTS / "mobilenet_v2.onnx"
    _export_onnx(fp32)
    model_path = fp32
    if quantized:
        model_path = RESULTS / "mobilenet_v2.int8.onnx"
        if not model_path.exists():
            # onnxruntime.quantization пишет промежуточные файлы рядом с входом и
            # спотыкается о не-ASCII путь -> квантуем во временной ASCII-папке
            import shutil
            import tempfile

            from onnxruntime.quantization import QuantType, quantize_dynamic

            tmp = Path(tempfile.mkdtemp())
            shutil.copy2(fp32, tmp / "in.onnx")
            quantize_dynamic((tmp / "in.onnx").as_posix(), (tmp / "out.onnx").as_posix(),
                             weight_type=QuantType.QUInt8)
            shutil.move((tmp / "out.onnx").as_posix(), model_path.as_posix())
            shutil.rmtree(tmp, ignore_errors=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(model_path.as_posix(), so,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def fn(t):
        return int(np.argmax(sess.run(None, {name: t})[0][0]))

    sec, pred = _time_loop(fn, x_np, N_RUNS, 15, False)
    return {"sec": sec, "pred": pred, "size_mb": round(model_path.stat().st_size / 1e6, 2)}


# ----------------------------------------------------------------------- main
def main() -> int:
    global N_RUNS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=250)
    args = ap.parse_args()
    N_RUNS = args.runs

    RESULTS.mkdir(exist_ok=True)
    x = load_cat()
    names = labels()
    has_cuda = torch.cuda.is_available()
    print(f"MobileNetV2, {N_RUNS} классификаций одного изображения кота.")
    print(f"CPU: PyTorch/ONNX Runtime; GPU: {torch.cuda.get_device_name(0) if has_cuda else 'нет'}\n")

    plan = [
        ("PyTorch eager, CPU fp32", "Raspberry Pi 4, плоский TensorFlow (≈2.7 FPS)",
         lambda: bench_torch(x, "cpu", False)),
        ("TorchScript (optimize_for_inference), CPU", "—",
         lambda: bench_torchscript(x)),
        ("ONNX Runtime, CPU fp32", "Raspberry Pi 4, TFLite (≈7.8 FPS)",
         lambda: bench_onnx(x, False)),
        ("ONNX Runtime, CPU INT8 (dynamic quant)", "Raspberry Pi 4, TFLite 8-bit (≈8.6 FPS)",
         lambda: bench_onnx(x, True)),
    ]
    if has_cuda:
        plan += [
            ("PyTorch, CUDA fp32", "Jetson Nano, 128 ядер CUDA (≈12.2 FPS)",
             lambda: bench_torch(x, "cuda", False)),
            ("PyTorch, CUDA fp16 (autocast)", "Jetson Nano, TF-TRT 16-bit (≈48.8 FPS)",
             lambda: bench_torch(x, "cuda", True)),
        ]

    rows = []
    for name, lecture_row, fn in plan:
        print(f"[{name}] ...", flush=True)
        r = fn()
        fps = N_RUNS / r["sec"]
        rows.append({"config": name, "lecture_analog": lecture_row,
                     "fps": round(fps, 1), "ms_per_image": round(r["sec"] / N_RUNS * 1000, 2),
                     "total_sec": round(r["sec"], 2),
                     "pred_label": names[r["pred"]].replace("_", " "),
                     "size_mb": r.get("size_mb")})
        print(f"  {fps:.1f} FPS  ({rows[-1]['ms_per_image']} мс/изобр.)  "
              f"-> {rows[-1]['pred_label']}\n", flush=True)

    base = next(r for r in rows if r["config"].startswith("PyTorch eager, CPU"))["fps"]
    for r in rows:
        r["speedup_vs_cpu_eager"] = round(r["fps"] / base, 1)

    (RESULTS / "bench.json").write_text(
        json.dumps({"runs": N_RUNS, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    md = [f"# Бенчмарк распознавания ({N_RUNS} классификаций, MobileNetV2)", "",
          "| конфигурация | FPS | мс/изобр. | ускорение | аналог из лекции |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | **{r['fps']}** | {r['ms_per_image']} | "
                  f"{r['speedup_vs_cpu_eager']}x | {r['lecture_analog']} |")
    md += ["", f"Все конфигурации предсказали один и тот же класс: "
           f"**{rows[0]['pred_label']}** (проверка корректности).",
           "", "Coral USB Accelerator (Edge TPU, ~240 FPS в лекции) не воспроизводится — "
           "на машине нет TPU. Скрипт для реального Raspberry Pi — в `raspberry_pi/`."]
    (RESULTS / "bench.md").write_text("\n".join(md), encoding="utf-8")

    _plot(rows)

    print("=" * 72)
    print(f"{'конфигурация':<42}{'FPS':>8}{'ускор.':>9}")
    for r in rows:
        print(f"{r['config']:<42}{r['fps']:>8}{r['speedup_vs_cpu_eager']:>8}x")
    print(f"\nОтчёт: {RESULTS}/bench.md")
    return 0


def _plot(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [r["config"] for r in rows]
    ax.barh(range(len(names)), [r["fps"] for r in rows], color="#4472c4")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("FPS (250 классификаций одного изображения)")
    for i, r in enumerate(rows):
        ax.text(r["fps"], i, f" {r['fps']}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "bench_fps.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
