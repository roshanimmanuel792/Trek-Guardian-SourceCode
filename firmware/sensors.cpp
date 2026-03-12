#include "sensors.h"
#include "config.h"

#include <Wire.h>
#include <math.h>
#include <Adafruit_BME280.h>
#include <DHT.h>
#include <TinyGPS++.h>
#include <MAX30105.h>
#include "spo2_algorithm.h"

Adafruit_BME280 bme;
DHT dht(DHT_PIN, DHT11);
MAX30105 particleSensor;
TinyGPSPlus gps;

uint32_t irBuffer[100];
uint32_t redBuffer[100];

SensorData data;

void initSensors() {

  Wire.begin();

  if(!bme.begin(0x76))
    Serial.println("BME280 not detected");

  dht.begin();

  if(!particleSensor.begin(Wire))
    Serial.println("MAX30102 not detected");

  particleSensor.setup();
}

SensorData readSensors() {

  data.temperature = dht.readTemperature();
  data.humidity = dht.readHumidity();

  float pressure = bme.readPressure() / 100.0;

  data.altitude = 44330 * (1 - pow((pressure / SEA_LEVEL_PRESSURE),0.1903));

  for(byte i=0;i<100;i++){
      while(!particleSensor.available())
          particleSensor.check();

      redBuffer[i] = particleSensor.getRed();
      irBuffer[i] = particleSensor.getIR();

      particleSensor.nextSample();
  }

  int32_t spo2;
  int8_t spo2Valid;
  int32_t heartRate;
  int8_t hrValid;

  maxim_heart_rate_and_oxygen_saturation(
      irBuffer,100,redBuffer,
      &spo2,&spo2Valid,&heartRate,&hrValid
  );

  if(spo2Valid)
    data.spo2 = spo2;

  if(hrValid)
    data.heartRate = heartRate;

  while(Serial.available())
      gps.encode(Serial.read());

  if(gps.location.isValid()){
    data.latitude = gps.location.lat();
    data.longitude = gps.location.lng();
  }

  return data;
}
