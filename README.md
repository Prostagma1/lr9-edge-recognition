# ЛР9. Распознавание образов на одноплатном устройстве

## Задание

> Используя исходные коды и примеры, приведённые в лекции, реализуйте программу по
> распознаванию образов на одноплатном устройстве.

## Что сделано

Одноплатного устройства (Raspberry Pi / Jetson / Arduino) под рукой нет, поэтому
сделаны программы **для запуска на самих платах** + воспроизведение бенчмарка из
лекции на доступном железе:

1. **`raspberry_pi/classify_pi.py`** — программа **для Raspberry Pi**: воспроизводит
   бенчмарк из лекции (250 классификаций одного изображения на MobileNetV2 через
   `tflite_runtime`), режим непрерывного распознавания с камеры Pi (`--camera`),
   поддержка Coral Edge TPU (`--edgetpu`). Установка — `raspberry_pi/README.md`.

2. **`arduino/`** — программы **для Arduino Uno** (ATmega328P, 2 КБ SRAM). MobileNetV2
   и TFLite Micro туда не влезут, поэтому «образ» ужат до минимума, а инференс —
   прямой проход int8-сети на чистом C. **Камера не нужна** ни в одном варианте:
   * `digit_recognizer/` — рукописная цифра **8×8** (`sklearn digits`), сеть
     64→16→10, int8 acc **0.969** (= float32). Картинка либо зашита, либо приходит
     по Serial (`uv run lr9-send-digit`).
   * `signal_shape/` — форма аналогового сигнала на входе **A0** (32 отсчёта):
     sine/square/triangle/sawtooth/flat/noise, сеть 32→24→6, int8 acc ≈0.87.
     Данные — с потенциометра/генератора на A0 или по Serial (`uv run lr9-send-signal`).

   Веса генерируют `lr9-arduino-export` / `lr9-signal-export`. Подробно —
   `arduino/README.md`.

3. **`lr9-bench`** — тот же бенчмарк из лекции (250 классификаций, MobileNetV2),
   запущенный на доступном железе (ноутбук, RTX 3050 Ti) на разных бэкендах,
   сопоставленных с рядами таблиц из лекции.

## Проверка на эмуляции железа (GitHub Actions)

Workflow [`.github/workflows/edge.yml`](.github/workflows/edge.yml), последний
успешный прогон —
[actions/runs/38057747017](https://github.com/Prostagma1/lr9-edge-recognition/actions/runs/38057747017).

* **Arduino Uno в simavr.** `arduino-cli` собирает оба скетча под `arduino:avr:uno`,
  `.elf` исполняется в потактовом эмуляторе ATmega328P @ 16 МГц, UART — в лог.
* **Raspberry Pi-стек на ARM64.** Раннер `ubuntu-24.04-arm`, контейнер
  `arm64v8/debian:bookworm` (база Raspberry Pi OS), Python 3.11 + tflite-runtime 2.14,
  `classify_pi.py --runs 250` на моделях из лекции.

| платформа | задача | результат | скорость | ресурсы |
|---|---|---|---|---|
| Uno (simavr) | цифра 8×8 | 10/10 | 5668 мкс | flash 5294 Б (16 %), SRAM 268 Б (13 %) |
| Uno (simavr) | форма сигнала | 6/6 | 4682 мкс | flash 4814 Б (14 %), SRAM 296 Б (14 %) |
| ARM64 Neoverse-N2 | MobileNetV2 8-bit | Egyptian cat | 153.4 FPS | Pi 4 в лекции: 8.6 FPS |
| ARM64 Neoverse-N2 | MobileNetV2 float32 | Egyptian cat | 238.9 FPS | Pi 4 в лекции: 7.8 FPS |

ARM64-раннер — серверное ядро, заметно быстрее Cortex-A72 у Pi 4: прогон подтверждает
архитектуру и программный стек, а не скорость самой платы.

![run](screenshots/run.png)
![uno digit](screenshots/uno_digit.png)
![uno signal](screenshots/uno_signal.png)
![pi bench](screenshots/pi_bench.png)

## Как запустить

```bash
uv sync
uv run lr9-bench                         # 250 классификаций MobileNetV2 на всех бэкендах
uv run lr9-bench --runs 500

uv run lr9-arduino-export                # веса цифры 8x8   -> arduino/digit_recognizer/*.h
uv run lr9-signal-export                 # веса формы сигнала -> arduino/signal_shape/*.h
uv run lr9-send-digit  --sweep           # сухой прогон (без платы); с --port COM3 — на плату
uv run lr9-send-signal --sweep
```

## Результаты (250 классификаций, MobileNetV2, изображение кота)

| конфигурация | FPS | мс/изобр. | ускорение | аналог из лекции (её FPS) |
|---|---|---|---|---|
| PyTorch eager, CPU fp32 | 76.7 | 13.0 | 1.0× | Raspberry Pi 4, плоский TensorFlow (2.7) |
| TorchScript `optimize_for_inference`, CPU | 145.0 | 6.9 | 1.9× | — (аналог `@tf.function`) |
| **ONNX Runtime, CPU fp32** | **363.1** | **2.8** | **4.7×** | Raspberry Pi 4, TFLite (7.8) |
| ONNX Runtime, CPU INT8 (dynamic quant) | 48.1 | 20.8 | 0.6× | Raspberry Pi 4, TFLite 8-bit (8.6) |
| PyTorch, CUDA fp32 | 122.9 | 8.1 | 1.6× | Jetson Nano, 128 ядер CUDA (12.2) |
| PyTorch, CUDA fp16 (autocast) | 98.2 | 10.2 | 1.3× | Jetson Nano, TF-TRT 16-bit (48.8) |

Все конфигурации выдали один и тот же класс — **Egyptian cat** (тот же результат,
что в лекции 1 и ЛР1). График — `results/bench_fps.png`.

## Выводы

1. **Оптимизированный рантайм — главный рычаг, ровно как в лекции.** Переход с
   «наивного» PyTorch на ONNX Runtime дал **×4.7** на CPU — это тот же эффект, что в
   лекции даёт переход с плоского TensorFlow на TFLite на Raspberry Pi (там ×2.9:
   2.7 → 7.8 FPS). Графовая оптимизация (TorchScript) — промежуточный вариант, ×1.9.

2. **8-битное квантование ускоряет НЕ всегда.** На Raspberry Pi (ARM) в лекции INT8
   дал прибавку (7.8 → 8.6 FPS): у ARM есть быстрые INT8-инструкции, а TFLite —
   рантайм, собранный под них. На десктопном x86 наивное *динамическое* квантование
   ONNX, наоборот, **замедлило в 1.6 раза** (48 против 363 FPS): свёрточные INT8-ядра
   без калибровки и QDQ-графа работают хуже оптимизированных FP32, а накладные
   расходы на quant/dequant на каждом слое съедают выигрыш. Вывод: эффект квантования
   зависит от связки «железо + рантайм», его надо **измерять**, а не предполагать —
   это и есть главный методический урок лекции.

3. **GPU на batch = 1 почти не помогает** (та же картина, что в ЛР1 и ЛР4): CUDA fp32
   (123 FPS) медленнее ONNX-CPU (363 FPS), потому что при одном изображении время
   уходит на запуск ядер и пересылку данных CPU↔GPU, а не на вычисления. fp16 при
   batch = 1 ещё и медленнее fp32 (autocast-накладные). На Jetson выигрыш от TF-TRT
   (12 → 49 FPS) виден, потому что там нет быстрого CPU-альтернативы и TensorRT
   компилирует специализированный движок.

4. **Coral USB (Edge TPU)** в лекции даёт ~240 FPS — это специализированная микросхема
   под 8-битные операции. Аналога на ноутбуке нет; для реального Pi путь к этим
   числам — `classify_pi.py --edgetpu` с моделью, скомпилированной под Edge TPU.

5. **Чем меньше устройство — тем радикальнее ужимается всё.** На Arduino Uno (2 КБ
   SRAM) не помещается даже интерпретатор TFLite Micro, поэтому пришлось спуститься
   от 224×224 RGB + MobileNetV2 до **8×8 grayscale + MLP на 1210 параметров**, а
   инференс написать руками на C. При этом int8-квантование здесь не «опция ради
   скорости», а необходимость: float32-веса + программная эмуляция float на 8-битном
   AVR были бы тяжелы, а int8 даёт 1.9 КБ во flash и целочисленную арифметику —
   **без потери точности** (0.969 = как у float32).

## Структура

```
lr9/
├── src/lr9/
│   ├── benchmark.py         # бенчмарк из лекции на доступных бэкендах (MobileNetV2, 250x)
│   ├── tiny_mlp.py          # общий код int8-MLP: квантование, forward_int8, генерация .h
│   ├── arduino_export.py    # цифра 8x8: обучение + int8, экспорт весов
│   ├── signal_export.py     # форма сигнала на A0: обучение + int8, экспорт весов
│   └── host_tools.py        # lr9-send-digit / lr9-send-signal: кормят плату по Serial
├── raspberry_pi/
│   ├── classify_pi.py       # программа ДЛЯ Raspberry Pi (tflite_runtime + камера + Coral)
│   └── README.md
├── arduino/
│   ├── digit_recognizer/    # .ino + model_data.h + test_images.h (цифра 8x8)
│   ├── signal_shape/        # .ino + signal_model.h + signal_labels.h + test_signals.h (A0)
│   └── README.md            # сборка/прошивка, проверка без платы, ожидаемый Serial-вывод
└── results/                 # bench.md, bench.json, bench_fps.png
```
