#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

#include "config.h"
#include "sensors.h"
#include "ml_model.h"
#include "captive_portal.h"

SensorData data;

// Causal EWMA state for firmware features (must match training EWMA_ALPHA)
float spo2Ewma = 0;
float hrEwma = 0;
bool ewmaInitialized = false;
float prevSpo2Ewma = 0;
float prevHrEwma = 0;

void updateEwma(float spo2, float heartRate) {
  if (!ewmaInitialized) {
    spo2Ewma = spo2;
    hrEwma = heartRate;
    prevSpo2Ewma = spo2;
    prevHrEwma = heartRate;
    ewmaInitialized = true;
    return;
  }
  prevSpo2Ewma = spo2Ewma;
  prevHrEwma = hrEwma;
  spo2Ewma = EWMA_ALPHA * spo2 + (1.0f - EWMA_ALPHA) * spo2Ewma;
  hrEwma = EWMA_ALPHA * heartRate + (1.0f - EWMA_ALPHA) * hrEwma;
}

void setup() {

  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  initSensors();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

}

void loop() {

  data = readSensors();
  updateEwma(data.spo2, data.heartRate);

  // Approximate slopes from EWMA deltas between loops (~2s)
  float spo2Slope = spo2Ewma - prevSpo2Ewma;
  float hrSlope = hrEwma - prevHrEwma;

  String risk = predictRisk(
    data.altitude,
    data.spo2,
    data.heartRate,
    spo2Ewma,
    hrEwma,
    spo2Slope,
    hrSlope
  );

  if (risk == "HIGH" || risk == "CRITICAL") {

    digitalWrite(LED_PIN, HIGH);
    tone(BUZZER_PIN, 1000);

    if (WiFi.status() == WL_CONNECTED)
      sendCloudAlert(data, risk);
    else {
      startCaptivePortal(data.spo2, data.altitude, risk);
      handlePortal();
    }

  }

  else {
    digitalWrite(LED_PIN, LOW);
    noTone(BUZZER_PIN);
  }

  delay(2000);
}

void sendCloudAlert(SensorData data, String risk) {

  WiFiClient client;
  HTTPClient http;

  http.begin(client, ALERT_SERVER);
  http.addHeader("Content-Type", "application/json");

  String payload = "{";
  payload += "\"spo2\":" + String(data.spo2) + ",";
  payload += "\"altitude\":" + String(data.altitude) + ",";
  payload += "\"heartRate\":" + String(data.heartRate) + ",";
  payload += "\"spo2_ewma\":" + String(spo2Ewma) + ",";
  payload += "\"hr_ewma\":" + String(hrEwma) + ",";
  payload += "\"spo2_slope\":" + String(spo2Ewma - prevSpo2Ewma) + ",";
  payload += "\"hr_slope\":" + String(hrEwma - prevHrEwma) + ",";
  payload += "\"lat\":" + String(data.latitude) + ",";
  payload += "\"lon\":" + String(data.longitude) + ",";
  payload += "\"risk\":\"" + risk + "\"";
  payload += "}";

  http.POST(payload);
  http.end();
}
