#ifndef CONFIG_H
#define CONFIG_H

#define DEVICE_NAME "Trek-Guardian"
#define FIRMWARE_VERSION "1.0.0"

#define WIFI_SSID "TrekGuardian"
#define WIFI_PASSWORD "trek1234"

#define SERVER_HOST "your-server.com"
#define SERVER_PORT 80
#define API_ENDPOINT "/api/alerts"

#define SENSOR_READ_INTERVAL 5000
#define ALERT_THRESHOLD_SPO2 90
#define ALERT_THRESHOLD_ALTITUDE 3000

#define LED_PIN 2
#define BUZZER_PIN 4
#define DHT_PIN 0

#define SEA_LEVEL_PRESSURE 1013.25

#define DEEPSLEEP_INTERVAL 300000

#define ALERT_SERVER "http://your-server.com/api/alerts"

#endif
