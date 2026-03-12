#ifndef SENSORS_H
#define SENSORS_H

struct SensorData {
  float spo2;
  float heartRate;
  float altitude;
  float temperature;
  float humidity;
  float latitude;
  float longitude;
};

void initSensors();
SensorData readSensors();

#endif
