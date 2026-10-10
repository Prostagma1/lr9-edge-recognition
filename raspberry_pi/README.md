# Распознавание на Raspberry Pi

`classify_pi.py` — программа для запуска **на самом Raspberry Pi** (проверялось на
Pi 4 / Pi 5, Raspberry Pi OS Bookworm). Повторяет бенчмарк из лекции 9 и умеет
работать с камерой.

## Установка на Pi

```bash
sudo apt update && sudo apt install -y python3-pip python3-picamera2
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install pillow numpy
pip install tflite-runtime            # если нет — pip install ai-edge-litert

# модель + метки (квантованная MobileNetV2, ImageNet):
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar xzf mobilenet_v2_1.0_224_quant.tgz
curl -o labels.txt https://raw.githubusercontent.com/google-coral/test_data/master/imagenet_labels.txt
cp /usr/share/rpicam-apps/imx708.json . 2>/dev/null || true   # не обязательно
# тестовая картинка:
wget -O cat.jpg https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02123045_tabby.JPEG
```

Для Coral USB Accelerator дополнительно:
```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update && sudo apt install -y libedgetpu1-std
# и модель, скомпилированную под Edge TPU:
wget https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite
```

## Запуск

```bash
# бенчмарк (как в лекции: 250 классификаций одного изображения)
python3 classify_pi.py --runs 250

# то же на float-модели
python3 classify_pi.py --model mobilenet_v2_1.0_224.tflite --float

# на Coral Edge TPU
python3 classify_pi.py --model mobilenet_v2_1.0_224_quant_edgetpu.tflite --edgetpu

# непрерывное распознавание с камеры Pi
python3 classify_pi.py --camera
```

## Ожидаемые значения FPS (из лекции)

| конфигурация | FPS |
|---|---|
| Raspberry Pi 4, плоский TensorFlow | 2.7 |
| Raspberry Pi 4, TFLite | 7.8 |
| Raspberry Pi 4, TFLite 8-bit | 8.6 |
| Raspberry Pi 3B+, TFLite 8-bit | 4.2 |
| Raspberry Pi 4 + Coral USB (USB 3) | ~240 |

На ПК без Pi-железа эквивалентный бенчмарк на других бэкендах (ONNX Runtime, CUDA)
запускается через `uv run lr9-bench` в корне проекта — см. основной README.
