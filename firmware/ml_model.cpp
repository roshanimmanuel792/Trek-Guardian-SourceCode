#include "ml_model.h"

String predictRisk(float altitude, float spo2, float heartRate){

  if(spo2 < 88 && altitude > 3800)
    return "CRITICAL";

  if(spo2 < 90 && altitude > 3200)
    return "HIGH";

  if(spo2 < 93 && altitude > 2500)
    return "MODERATE";

  return "LOW";
}
