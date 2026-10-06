# Plant Pathogen Detection and Targeted Spraying

This project detects maize leaf diseases from images uploaded by an ESP32-CAM or the web interface. It records detections and devices in SQLite. An authenticated camera can trigger a spray command after repeated high-confidence detections; a separate ESP32 controller polls for queued commands and drives three relay outputs.

## Components

- `main.py` — FastAPI service, image preprocessing and classification, SQLite history, and spray command queue.
- `templates/index.html` — browser interface (served by the application).
- `Node/Node.ino` — ESP32-CAM firmware. Captures a JPEG every 30 seconds and posts it to the prediction endpoint.
- `PumpController/PumpController.ino` — separate ESP32 DevKit firmware. Polls for commands and controls three relay channels.
- `Node/secrets.h.example` and `PumpController/secrets.h.example` — configuration templates. Copy each to `secrets.h` and fill in local values; `secrets.h` is ignored by Git.

## Disease classes and treatment mapping

| Model class | Pump | Configured tank label |
| --- | ---: | --- |
| `Blight` | 1 | copper fungicide |
| `Common_Rust` | 2 | sulfur fungicide |
| `Gray_Leaf_Spot` | 3 | triazole fungicide |
| `Healthy` | — | No spray |

The tank labels describe the configured mapping only. Use appropriate treatments and follow their product labels and local guidance.

## Run the API

Python dependencies are listed in `requirements.txt`. Install them in a Python environment, then start the service with:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first startup, the application downloads its model from the configured Google Drive file if `plant_disease_model.h5` is not present. The model file and SQLite database are local runtime data and are not committed.

Set `DEVICE_API_KEY` in the server environment to a long, private key. Configure the same value as `API_KEY` in both device `secrets.h` files. Without a matching server key, device uploads will still be classified and recorded, but they will not create spray commands; `/pump/next` rejects requests.

## Flash the devices

For each firmware folder, copy its example configuration and edit it:

```bash
cp Node/secrets.h.example Node/secrets.h
cp PumpController/secrets.h.example PumpController/secrets.h
```

Set `WIFI_SSID`, `WIFI_PASS`, and `API_KEY` in both files. Open each `.ino` sketch in the Arduino IDE with ESP32 board support and the corresponding camera or ESP32 board selected. Flash `Node/Node.ino` to an ESP32-CAM and `PumpController/PumpController.ino` to a separate ESP32 DevKit. The camera device ID in both sketches must match (`ESP32-CAM-01` by default).

The pump controller assumes an active-low relay module wired to GPIO 25, 26, and 27. Verify your relay board's logic, wiring, and safe fluid handling before connecting pumps; change `RELAY_ON`, `RELAY_OFF`, or `PUMP_PINS` to match the hardware.

## API overview

- `POST /predict` — multipart fields `file`, optional `device_id` (defaults to `browser`), and optional `api_key`. Returns the prediction and an optional action. Browser scans do not trigger spraying.
- `GET /results` — recent detections; accepts optional `device_id` filtering.
- `GET /devices` — registered devices and last-seen times.
- `POST /pump/next` — form fields `device_id` and `api_key`; returns one pending action or `{"action": null}`. Commands expire after 60 seconds.
- `GET /sprays` — recent spray queue records.

The server only queues treatment for mapped disease classes when the latest two recorded readings for that device agree, each has confidence of at least 0.70, and both are within a three-minute window. It applies a ten-minute cooldown per device and pump. The configured spray duration is 3 seconds; the controller also caps any received duration at 5 seconds. The regular display confidence threshold is 0.45. No-leaf and uncertain images are not stored as classified results or considered for spray decisions.

SQLite data is stored in `results.db` in the process working directory. The database is ignored by Git; configure persistent storage in the hosting environment if history must survive restarts.
