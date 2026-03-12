#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

#include "config.h"
#include "sensors.h"
#include "ml_model.h"
#include "captive_portal.h"

SensorData data;

void setup(){

  Serial.begin(115200);

  pinMode(LED_PIN,OUTPUT);
  pinMode(BUZZER_PIN,OUTPUT);

  initSensors();

  WiFi.begin(WIFI_SSID,WIFI_PASSWORD);

}

void loop(){

  data = readSensors();

  String risk = predictRisk(data.altitude,data.spo2,data.heartRate);

  if(risk=="HIGH" || risk=="CRITICAL"){

    digitalWrite(LED_PIN,HIGH);
    tone(BUZZER_PIN,1000);

    if(WiFi.status()==WL_CONNECTED)
      sendCloudAlert(data,risk);
    else{
      startCaptivePortal(data.spo2,data.altitude,risk);
      handlePortal();
    }

  }

  else{
    digitalWrite(LED_PIN,LOW);
    noTone(BUZZER_PIN);
  }

  delay(2000);
}

void sendCloudAlert(SensorData data,String risk){

  HTTPClient http;

  http.begin(ALERT_SERVER);

  http.addHeader("Content-Type","application/json");

  String payload="{";
  payload+="\"spo2\":"+String(data.spo2)+",";
  payload+="\"altitude\":"+String(data.altitude)+",";
  payload+="\"heartRate\":"+String(data.heartRate)+",";
  payload+="\"lat\":"+String(data.latitude)+",";
  payload+="\"lon\":"+String(data.longitude)+",";
  payload+="\"risk\":\""+risk+"\"";
  payload+="}";

  http.POST(payload);
  http.end();
}
