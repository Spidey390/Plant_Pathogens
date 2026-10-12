# Plant Pathogen Detection and Targeted Spraying

This project detects maize leaf diseases from images uploaded by an IoT-enabled ESP32-CAM node or through the web dashboard. Images are classified using a deep learning model running on a FastAPI backend. When an authenticated node detects a high-confidence plant pathogen, it automatically actuates a dedicated relay-driven pump to spray the appropriate targeted treatment directly.

---

## Hardware Components

The automated detection and spraying system consists of the following hardware modules:

| Component | Specification / Model | Purpose / Details |
| :--- | :--- | :--- |
| **Microcontroller & Vision Unit** | **ESP32-CAM (AI-Thinker)** | Dual-core 32-bit MCU with built-in 2.4 GHz Wi-Fi, 4 MB external PSRAM, and internal DRAM. Captures leaf imagery, negotiates HTTPS with the API, and drives the spraying relays. |
| **Camera Sensor** | **OV2640 (2 Megapixel)** | Connected directly to the ESP32-CAM DVP interface. Configured for JPEG capture at QVGA (320x240) resolution with `CAMERA_GRAB_LATEST` buffer mode. |
| **Actuator Controller** | **3-Channel (or 4-Channel) 5V Relay Module** | Optocoupler-isolated, active-LOW relay board (`RELAY_ON = LOW`, `RELAY_OFF = HIGH`). Controls power switching to individual spray pumps while isolating the MCU from inductive motor spikes. |
| **Fluid Dispensing Pumps** | **3x Mini DC Pumps (5V or 12V)** | Submersible or diaphragm micro-pumps dedicated to individual fungicide tanks. |
| **Chemical Storage Tanks** | **3x Liquid Reservoirs** | - **Tank 1:** Copper fungicide (for Blight)<br>- **Tank 2:** Sulfur fungicide (for Common Rust)<br>- **Tank 3:** Triazole fungicide (for Gray Leaf Spot) |
| **Plumbing & Atomization** | **Silicone Tubing & Misting Nozzles** | Distributes pressurized fluid from pumps to atomized sprayers directed toward the target crop area. |
| **MCU & Logic Power Supply** | **5V DC Power Adapter (≥ 2.0A)** | Powers the ESP32-CAM (via 5V & GND) and relay board logic. A dedicated, high-amperage 5V supply is required to prevent brownout resets during Wi-Fi transmission bursts. |
| **Pump Motor Power Supply** | **External DC Power Source (5V/12V)** | Supplies power to the pump motors through the relay Common (COM) and Normally Open (NO) terminals. Kept electrically separate from the MCU logic rail. |
| **Flashing / Programming Interface** | **FTDI USB-to-TTL Adapter / ESP32-CAM-MB Shield** | Required to program the ESP32-CAM (via U0TXD, U0RXD, and GPIO 0 strapped to GND during flashing). |

---

## Hardware Wiring & Pin Mapping

### ESP32-CAM to 3-Channel Relay Module

| ESP32-CAM Pin | Relay Module Pin | Assigned Disease Class | Targeted Treatment |
| :--- | :--- | :--- | :--- |
| **GPIO 13** | **IN 1** | `Blight` | Pump 1: Copper Fungicide |
| **GPIO 14** | **IN 2** | `Common_Rust` | Pump 2: Sulfur Fungicide |
| **GPIO 15** | **IN 3** | `Gray_Leaf_Spot` | Pump 3: Triazole Fungicide |
| **5V** | **VCC** *(Logic)* | — | Logic reference |
| **GND** | **GND** | — | Common Ground |

> [!IMPORTANT]
> **Power Isolation & Brownout Prevention:**
> 1. **Do not power the pump motors from the ESP32-CAM 5V pin.** Relay coil switching and motor current draw will cause severe voltage dips and trigger the ESP32 brownout detector.
> 2. Connect the relay module's **JD-VCC** jumper to an independent 5V power supply, and ensure all grounds (GND) are securely tied together.
> 3. Avoid using **GPIO 16** on the ESP32-CAM as it is internally reserved for external PSRAM chip select (CS).

---

## Disease Classes and Treatment Mapping

| Model Class | Pump Output | Configured Tank Treatment |
| :--- | :---: | :--- |
| `Blight` | **Pump 1 (GPIO 13)** | Copper fungicide |
| `Common_Rust` | **Pump 2 (GPIO 14)** | Sulfur fungicide |
| `Gray_Leaf_Spot` | **Pump 3 (GPIO 15)** | Triazole fungicide |
| `Healthy` | — | No spray |

---

## System Architecture & Software Components

- **`main.py`** — FastAPI application providing model inference, HSV green-leaf validation, prediction logging, SQLite persistence, and bench testing (`/pump/test`).
- **`templates/index.html`** — Real-time monitoring dashboard and browser testing interface.
- **`Node/Node.ino`** — Integrated ESP32-CAM firmware:
  - Captures JPEG image via OV2640.
  - Uploads image and device credentials to `/predict`.
  - Parses prediction and triggers the respective relay channel directly for the configured spray duration (default: 5 seconds).
- **`Node/secrets.h.example`** — Configuration template for Wi-Fi credentials (`WIFI_SSID`, `WIFI_PASS`) and `API_KEY`.

---

## Setup & Running the API

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI backend:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   *On first run, the deep learning model (`plant_disease_model.h5`) will automatically be downloaded from Google Drive if not present.*

3. **Configure API Security:**
   Set the `DEVICE_API_KEY` environment variable in your production environment. Configure the same key as `API_KEY` in `Node/secrets.h`.

---

## Flashing the ESP32-CAM

1. Copy the configuration template:
   ```bash
   cp Node/secrets.h.example Node/secrets.h
   ```
2. Edit `Node/secrets.h` with your Wi-Fi SSID, password, and `API_KEY`.
3. Open `Node/Node.ino` in the Arduino IDE:
   - **Board:** `AI Thinker ESP32-CAM`
   - **CPU Frequency:** `240MHz`
   - **Flash Frequency:** `80MHz`
   - **Flash Mode:** `QIO`
   - **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)`
   - **PSRAM:** `Enabled`
4. Connect the FTDI programmer (Bridge **GPIO 0 to GND** before powering on to enter flashing mode).
5. Flash the sketch, disconnect GPIO 0 from GND, and press the Reset button.

---

## API Overview

- `POST /predict` — Multipart form upload (`file`, `device_id`, `api_key`). Performs HSV leaf detection, runs model inference, returns predicted class and confidence.
- `POST /pump/test` — Bench-testing endpoint (`device_id`, `pump`, `duration_ms`, `api_key`).
- `GET /results` — Returns recent classification history (supports `device_id` query parameter).
- `GET /devices` — Returns registered devices and online/offline status.
- `GET /sprays` — Returns log of recent spray events.
- `GET /` — Serves the interactive web monitoring dashboard.
