# Trek Guardian

<p align="center">
  <img src="https://img.shields.io/badge/ESP8266-Firmware-green?style=for-the-badge&logo=espressif" alt="ESP8266">
  <img src="https://img.shields.io/badge/Python-Flask-blue?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Hardware-IoT-orange?style=for-the-badge&logo=arduino" alt="IoT">
</p>

> Real-time altitude and SpO2 monitoring system for high-altitude trekkers with emergency alert capabilities

---

## Overview

**Trek Guardian** is an IoT+ML based health monitoring system designed for mountaineers and trekkers operating in high-altitude environments. The device continuously monitors vital health metrics including blood oxygen saturation (SpO2), heart rate, altitude, temperature, humidity, and GPS location. When dangerous conditions are detected, it triggers immediate alerts via cloud API or establishes a local captive portal for rescue coordination which doesnt require network.


KINDLY REFER THE https://github.com/roshanimmanuel792/Trek-Guardian.git FOR INSTALLATION OF DATASETS

---

## Features

- **Continuous Vital Monitoring** — Real-time SpO2, heart rate, altitude, temperature, and humidity tracking
- **Risk Assessment Engine** — Decision Tree ML model trained on altitude-SpO2 datasets for accurate risk classification
- **Dual Alert Mechanisms** — Cloud-based alerts via HTTP POST and offline captive portal fallback
- **Captive Portal Rescue Mode** — Self-hosted WiFi access point displays vital data when offline
- **GPS Location Tracking** — TinyGPS++ integration for precise coordinates in emergencies
- **Visual & Audio Alerts** — LED indicator and buzzer for immediate user feedback
- **Deep Sleep Optimization** — Configurable power-saving intervals for extended battery life

---

## Hardware Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | ESP8266 | WiFi-enabled processing unit |
| SpO2/Heart Rate | MAX30102 | Pulse oximetry & heart rate sensor |
| Pressure/Altitude | BME280 | Barometric pressure & temperature |
| Humidity | DHT11 | Ambient humidity detection |
| GPS Module | NEO-6M | Satellite positioning |
| Indicator | LED + Buzzer | Visual/audio alerts |

### Pin Configuration

```
ESP8266 Pin Mapping:
├── D1 (GPIO5)  → SDA (BME280)
├── D2 (GPIO4)  → SCL (BME280)
├── D3 (GPIO0)  → DHT11 Data
├── D5 (GPIO14) → MAX30102 IRQ
├── RX → GPS TX
├── TX → GPS RX
├── GPIO2 → LED
└── GPIO4 → Buzzer
```

---

## Software Architecture

```
Trek-Guardian_SourceCode/
├── firmware/
│   ├── trek_guardian_main.ino    # Main application loop
│   ├── config.h                  # System configuration
│   ├── sensors.h / .cpp         # Sensor abstraction layer
│   ├── ml_model.h / .cpp        # Risk prediction engine
│   └── captive_portal.h / .cpp  # Offline rescue portal
├── iot_server/
│   └── alert_api_example.py     # Flask REST API endpoint
└── docs/
    └── system_architecture.md   # Detailed architecture docs
```

---

## Getting Started

### Prerequisites

**Hardware:**
- ESP8266 (NodeMCU or Wemos D1 Mini)
- MAX30102 pulse oximeter
- BME280 barometric sensor
- DHT11 temperature/humidity sensor
- NEO-6M GPS module
- LED, Buzzer, Connecting wires

**Software:**
- Arduino IDE with ESP8266 board support
- Python 3.8+ (for IoT server)

### Install Dependencies

**Arduino Libraries (via Library Manager):**
- Adafruit MAX3010x
- Adafruit BME280
- DHT sensor library
- TinyGPS++
- ESP8266HTTPClient

**Python Dependencies:**
```bash
pip install flask
```

### Configuration

Edit `firmware/config.h`:
```cpp
#define WIFI_SSID "YourNetwork"
#define WIFI_PASSWORD "YourPassword"
#define SERVER_HOST "your-server.com"
#define SERVER_PORT 80
#define API_ENDPOINT "/alert"
```

### Build & Upload

1. Open `trek_guardian_main.ino` in Arduino IDE
2. Select your ESP8266 board
3. Configure upload speed to 115200
4. Upload sketch

### Run IoT Server

```bash
cd iot_server
python alert_api_example.py
```

---

## Risk Classification Model

The onboard ML model uses a **Decision Tree** classifier trained on the `altitude_spo2_dataset.csv` dataset using scikit-learn. The model was trained to categorize trekkers into risk levels based on altitude, SpO2, and heart rate metrics.

### Training Pipeline

```python
# Typical training workflow
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load dataset
df = pd.read_csv('dataset/altitude_spo2_dataset.csv')

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Export rules to C++ (see ml_model/export_rules.py)
export_to_cpp(clf, feature_names)
```

The trained model's decision rules were exported to `ml_model.cpp` for lightweight embedded inference on the ESP8266.

| Risk Level | Condition |
|------------|-----------|
| **CRITICAL** | SpO2 < 88% AND Altitude > 3800m |
| **HIGH** | SpO2 < 90% AND Altitude > 3200m |
| **MODERATE** | SpO2 < 93% AND Altitude > 2500m |
| **LOW** | All other conditions |

---

## API Integration

### Alert Payload

```json
POST /alert
Content-Type: application/json

{
  "spo2": 85,
  "altitude": 4200,
  "heartRate": 110,
  "lat": 27.9881,
  "lon": 86.9250,
  "risk": "HIGH"
}
```

### Response

```json
{
  "status": "alert received"
}
```

---

## Captive Portal Mode

When WiFi connectivity fails, the device automatically starts a captive portal:

- SSID: `TrekGuardian_Rescue`
- Access Point IP: `192.168.4.1`
- Web interface displays: SpO2, Altitude, Risk Level

Rescuers can connect to this network without internet to view the trekker's vital data.

---

## Power Consumption

| Mode | Current | Duration |
|------|---------|----------|
| Active | ~80mA | 2 seconds |
| Deep Sleep | ~15µA | 5 minutes |
| Avg Current | ~0.5mA | Per cycle |

With a 2000mAh LiPo battery, the device can operate for ~150 days in deep sleep mode.

---

## Future Enhancements

- [ ] Train Decision Tree on larger datasets for improved accuracy
- [ ] Experiment with Random Forest for ensemble predictions
- [ ] Add model retraining pipeline with new trekker data
- [ ] LoRaWAN integration for remote area connectivity
- [ ] Solar power harvesting
- [ ] Mobile companion app
- [ ] SOS button for manual emergency trigger
- [ ] Multi-language captive portal support

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Maxim Integrated for the MAX30102 algorithm
- Adafruit Industries for sensor libraries
- High-altitude medicine research communities

---

<p align="center">
  <strong>Built with ❤️ for mountain explorers</strong><br>
  Stay safe. Trek smart.
</p>
