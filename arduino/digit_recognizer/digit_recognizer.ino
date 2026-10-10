/*
 * ЛР9 — распознавание образов на Arduino Uno (ATmega328P).
 *
 * Распознаёт рукописную цифру 8x8 (значения 0..16) крошечной int8-сетью
 * 64 -> 16 -> 10 прямым проходом на чистом C. Веса лежат во flash (PROGMEM),
 * поэтому в 2 КБ SRAM всё помещается с огромным запасом.
 *
 * Веса и тестовые картинки генерируются командой `uv run lr9-arduino-export`
 * (см. arduino/README.md). Здесь их только не трогаем.
 *
 * Serial 115200:
 *   - при старте прогоняет N_TEST встроенных изображений, печатает
 *     предсказание, истину и время инференса (мкс);
 *   - затем принимает по Serial ровно IN_DIM(=64) байт (значения 0..16) —
 *     новое изображение — и отвечает распознанной цифрой.
 */

#include "model_data.h"    // IN_DIM, HID_DIM, OUT_DIM, W1/BIAS1/W2/BIAS2, REQUANT_M1, REQUANT_SHIFT
#include "test_images.h"   // N_TEST, TEST_IMAGES, TEST_LABELS

static int8_t  x[IN_DIM];
static int8_t  h[HID_DIM];

static inline int32_t clamp_i(int32_t v, int32_t lo, int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Прямой проход int8-сети. Возвращает распознанный класс 0..9.
uint8_t infer(const int8_t *img) {
  // --- слой 1: acc(int32) = BIAS1 + img . W1 ;  ReLU ;  requant в int8 ---
  for (uint8_t j = 0; j < HID_DIM; j++) {
    int32_t acc = (int32_t)pgm_read_dword(&BIAS1[j]);
    const int8_t *w = &W1[(uint16_t)j * IN_DIM];
    for (uint8_t i = 0; i < IN_DIM; i++) {
      acc += (int32_t)img[i] * (int8_t)pgm_read_byte(&w[i]);
    }
    if (acc < 0) acc = 0;                                   // ReLU
    int64_t q = ((int64_t)acc * REQUANT_M1
                 + ((int64_t)1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT;
    h[j] = (int8_t)clamp_i((int32_t)q, 0, 127);
  }

  // --- слой 2: acc(int32) = BIAS2 + h . W2 ;  argmax ---
  int32_t best_val = -2147483647L;
  uint8_t best_k = 0;
  for (uint8_t k = 0; k < OUT_DIM; k++) {
    int32_t acc = (int32_t)pgm_read_dword(&BIAS2[k]);
    const int8_t *w = &W2[(uint16_t)k * HID_DIM];
    for (uint8_t j = 0; j < HID_DIM; j++) {
      acc += (int32_t)h[j] * (int8_t)pgm_read_byte(&w[j]);
    }
    if (acc > best_val) { best_val = acc; best_k = k; }
  }
  return best_k;
}

void run_builtin_tests() {
  Serial.println(F("=== встроенные тесты (по одной картинке на цифру) ==="));
  uint8_t correct = 0;
  uint32_t total_us = 0;

  for (uint8_t t = 0; t < N_TEST; t++) {
    for (uint8_t i = 0; i < IN_DIM; i++) {
      x[i] = (int8_t)pgm_read_byte(&TEST_IMAGES[t][i]);
    }
    uint32_t t0 = micros();
    uint8_t pred = infer(x);
    uint32_t dt = micros() - t0;
    total_us += dt;

    uint8_t truth = pgm_read_byte(&TEST_LABELS[t]);
    correct += (pred == truth);
    Serial.print(F("img ")); Serial.print(t);
    Serial.print(F(": pred=")); Serial.print(pred);
    Serial.print(F(" truth=")); Serial.print(truth);
    Serial.print(pred == truth ? F("  OK  ") : F("  x   "));
    Serial.print(dt); Serial.println(F(" us"));
  }

  Serial.print(F("итог: ")); Serial.print(correct); Serial.print('/');
  Serial.print(N_TEST);
  Serial.print(F("  ср. ")); Serial.print(total_us / N_TEST);
  Serial.print(F(" us/инференс  (~"));
  Serial.print(1000000UL / (total_us / N_TEST));
  Serial.println(F(" инференсов/с)"));
  Serial.println(F("--- пришлите 64 байта (0..16) для своей картинки ---"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }
  Serial.println(F("Arduino Uno digit recognizer (int8 MLP 64-16-10)"));
  run_builtin_tests();
}

void loop() {
  if (Serial.available() >= IN_DIM) {
    for (uint8_t i = 0; i < IN_DIM; i++) {
      int v = Serial.read();
      x[i] = (int8_t)clamp_i(v, 0, 16);
    }
    uint32_t t0 = micros();
    uint8_t pred = infer(x);
    uint32_t dt = micros() - t0;
    Serial.print(F("распознано: ")); Serial.print(pred);
    Serial.print(F("  (")); Serial.print(dt); Serial.println(F(" us)"));
  }
}
