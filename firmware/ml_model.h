/**
 * Trek Guardian - Embedded ML Model
 * Generated: 2026-08-03 18:22:49 UTC
 *
 * Early hypoxia warning — distilled Decision Tree.
 * Features must match firmware EWMA state in trek_guardian_main.ino.
 */

#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <Arduino.h>

#define MODEL_VERSION "2.0.0"
#define NUM_FEATURES 7
#define EWMA_ALPHA 0.15f

// Firmware features (order matters for documentation only):
// altitude, spo2, heartRate, spo2_ewma, hr_ewma, spo2_slope, hr_slope

String predictRisk(
  float altitude,
  float spo2,
  float heartRate,
  float spo2_ewma,
  float hr_ewma,
  float spo2_slope,
  float hr_slope
);

#endif
