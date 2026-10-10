/*
 * ЛР9 — распознавание образов на Arduino Uno БЕЗ камеры и датчиков.
 *
 * «Образ» = форма аналогового сигнала на входе A0: 32 отсчёта, нормированные в
 * 0..16, классифицируются int8-сетью 32 -> 24 -> 6 на классы
 * { sine, square, triangle, sawtooth, flat, noise }.
 *
 * Что подать на A0:
 *   - потенциометр (рукой -> ramp / triangle / flat);
 *   - генератор / аудио через делитель;
 *   - PWM-пин Uno через RC-фильтр (sine/square можно сгенерировать самой платой);
 *   - ничего (плавающий вход) -> обычно noise / 50 Гц наводка.
 * Либо вообще без железа: прислать 32 байта (0..16) по Serial
 *   (uv run lr9-send-signal --port COM3 --shape triangle).
 *
 * Веса — сгенерированы `uv run lr9-signal-export`.
 */

#include "signal_model.h"    // IN_DIM=32, HID_DIM, OUT_DIM=6, W1/BIAS1/W2/BIAS2, REQUANT_*
#include "signal_labels.h"   // SHAPE_NAMES[OUT_DIM]
#include "test_signals.h"    // N_TEST, TEST_SIGNALS, TEST_LABELS

const uint8_t A0_PIN       = A0;
const uint16_t SAMPLE_US   = 500;    // 32 * 500 us = 16 ms окно (~2 кГц выборка)

static int8_t x[IN_DIM];
static int8_t h[HID_DIM];

static inline int32_t clampi(int32_t v, int32_t lo, int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

uint8_t infer(const int8_t *sig) {
  for (uint8_t j = 0; j < HID_DIM; j++) {
    int32_t acc = (int32_t)pgm_read_dword(&BIAS1[j]);
    const int8_t *w = &W1[(uint16_t)j * IN_DIM];
    for (uint8_t i = 0; i < IN_DIM; i++)
      acc += (int32_t)sig[i] * (int8_t)pgm_read_byte(&w[i]);
    if (acc < 0) acc = 0;
    int64_t q = ((int64_t)acc * REQUANT_M1
                 + ((int64_t)1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT;
    h[j] = (int8_t)clampi((int32_t)q, 0, 127);
  }
  int32_t best = -2147483647L; uint8_t bk = 0;
  for (uint8_t k = 0; k < OUT_DIM; k++) {
    int32_t acc = (int32_t)pgm_read_dword(&BIAS2[k]);
    const int8_t *w = &W2[(uint16_t)k * HID_DIM];
    for (uint8_t j = 0; j < HID_DIM; j++)
      acc += (int32_t)h[j] * (int8_t)pgm_read_byte(&w[j]);
    if (acc > best) { best = acc; bk = k; }
  }
  return bk;
}

// снять 32 отсчёта с A0 и нормировать min-max в 0..16 (ровно как при обучении)
void sample_A0(int8_t *out) {
  uint16_t raw[IN_DIM];
  uint16_t lo = 1023, hi = 0;
  for (uint8_t i = 0; i < IN_DIM; i++) {
    raw[i] = analogRead(A0_PIN);
    if (raw[i] < lo) lo = raw[i];
    if (raw[i] > hi) hi = raw[i];
    delayMicroseconds(SAMPLE_US);
  }
  uint16_t span = (hi > lo) ? (hi - lo) : 1;
  for (uint8_t i = 0; i < IN_DIM; i++)
    out[i] = (int8_t)(((uint32_t)(raw[i] - lo) * 16) / span);
}

void run_builtin_tests() {
  Serial.println(F("=== встроенные тесты ==="));
  uint8_t ok = 0; uint32_t us = 0;
  for (uint8_t t = 0; t < N_TEST; t++) {
    for (uint8_t i = 0; i < IN_DIM; i++)
      x[i] = (int8_t)pgm_read_byte(&TEST_SIGNALS[t][i]);
    uint32_t t0 = micros();
    uint8_t p = infer(x);
    us += micros() - t0;
    uint8_t truth = pgm_read_byte(&TEST_LABELS[t]);
    ok += (p == truth);
    Serial.print(F("win ")); Serial.print(t);
    Serial.print(F(": pred=")); Serial.print(SHAPE_NAMES[p]);
    Serial.print(F(" truth=")); Serial.print(SHAPE_NAMES[truth]);
    Serial.println(p == truth ? F("  OK") : F("  x"));
  }
  Serial.print(F("итог: ")); Serial.print(ok); Serial.print('/'); Serial.print(N_TEST);
  Serial.print(F("  ср. ")); Serial.print(us / N_TEST); Serial.println(F(" us/инференс"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  Serial.println(F("Arduino Uno signal-shape recognizer (int8 MLP 32-24-6)"));
  run_builtin_tests();
  Serial.println(F("--- крутите потенциометр на A0, либо шлите 32 байта по Serial ---"));
}

void loop() {
  // приоритет — окно, присланное по Serial (для теста без железа)
  if (Serial.available() >= IN_DIM) {
    for (uint8_t i = 0; i < IN_DIM; i++)
      x[i] = (int8_t)clampi(Serial.read(), 0, 16);
  } else {
    sample_A0(x);
  }
  uint8_t p = infer(x);
  Serial.print(F("форма: ")); Serial.println(SHAPE_NAMES[p]);
  delay(400);
}
