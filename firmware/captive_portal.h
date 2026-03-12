#ifndef CAPTIVE_PORTAL_H
#define CAPTIVE_PORTAL_H

#include <Arduino.h>

void startCaptivePortal(float spo2, float altitude, String risk);
void handlePortal();

#endif
