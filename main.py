from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import sqlite3
import datetime
from datetime import timedelta
import io
import os
import gdown
from PIL import Image
import cv2
MODEL_PATH = "plant_disease_model.h5"
GDRIVE_FILE_ID = "13e9PTZCsu10q8l0tnRSmQjjL8Kh5h9pK"

CONFIDENCE_THRESHOLD = 0.45
LEAF_GREEN_RATIO_THRESHOLD = 0.10

DB_PATH = "results.db"
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
DEVICE_API_KEY = os.environ.get("DEVICE_API_KEY")  # set on Render; ESP32 sends same value

# disease -> pump/tank. Pump number = relay channel on ESP32.
TREATMENT_MAP = {
    "Blight":         {"pump": 1, "tank": "copper fungicide"},
    "Common_Rust":    {"pump": 2, "tank": "sulfur fungicide"},
    "Gray_Leaf_Spot": {"pump": 3, "tank": "triazole fungicide"},
}
SPRAY_MIN_CONF = 0.70      # stricter than display threshold
CONFIRM_COUNT = 2          # N consecutive same-disease readings before spraying
CONFIRM_WINDOW_S = 180     # ...within this window
SPRAY_DURATION_MS = 3000   # mist burst length
SPRAY_COOLDOWN_S = 600     # min gap between sprays per device+pump
SPRAY_TTL_S = 60           # queued command expires if controller doesn't fetch in time

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")
    else:
        print("Model already exists, skipping download.")

download_model()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

def init_db():
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        disease TEXT,
        confidence REAL,
        device_id TEXT DEFAULT 'unknown'
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        name TEXT,
        last_seen TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS sprays (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        device_id TEXT,
        disease TEXT,
        pump INTEGER,
        duration_ms INTEGER,
        delivered INTEGER DEFAULT 0
    )""")
    try:
        cur.execute("ALTER TABLE sprays ADD COLUMN delivered INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    con.commit()
    con.close()

init_db()

def detect_leaf(img_array):
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    h, w = img_array.shape[:2]
    roi = green_mask[h // 6:5 * h // 6, w // 6:5 * w // 6]
    green_ratio = np.sum(roi == 255) / roi.size
    return green_ratio > LEAF_GREEN_RATIO_THRESHOLD

def preprocess_image(img_array):
    h, w = img_array.shape[:2]
    # Reduce crop to capture more context
    x1, y1 = w // 12, h // 12
    x2, y2 = 11 * w // 12, 11 * h // 12
    crop = img_array[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    resized = cv2.resize(enhanced, (224, 224))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

def upsert_device(device_id: str):
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("SELECT device_id FROM devices WHERE device_id=?", (device_id,))
    exists = cur.fetchone()
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if exists:
        cur.execute("UPDATE devices SET last_seen=? WHERE device_id=?", (now, device_id))
    else:
        cur.execute("INSERT INTO devices (device_id, name, last_seen) VALUES (?,?,?)", (device_id, device_id, now))
    con.commit()
    con.close()

def decide_action(device_id: str, disease: str):
    """Return spray command or None. Debounce + cooldown. Logs spray when issued."""
    spec = TREATMENT_MAP.get(disease)
    if not spec:
        return None
    now = datetime.datetime.utcnow()
    since = (now - timedelta(seconds=CONFIRM_WINDOW_S)).strftime(TS_FMT)
    cool = (now - timedelta(seconds=SPRAY_COOLDOWN_S)).strftime(TS_FMT)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT disease, confidence FROM results WHERE device_id=? AND timestamp>=? "
                "ORDER BY id DESC LIMIT ?", (device_id, since, CONFIRM_COUNT))
    rows = cur.fetchall()
    if len(rows) < CONFIRM_COUNT or any(r[0] != disease or r[1] < SPRAY_MIN_CONF for r in rows):
        con.close()
        return None
    cur.execute("SELECT 1 FROM sprays WHERE device_id=? AND pump=? AND timestamp>=? LIMIT 1",
                (device_id, spec["pump"], cool))
    if cur.fetchone():
        con.close()
        return None
    cur.execute("INSERT INTO sprays (timestamp, device_id, disease, pump, duration_ms) VALUES (?,?,?,?,?)",
                (now.strftime(TS_FMT), device_id, disease, spec["pump"], SPRAY_DURATION_MS))
    con.commit()
    con.close()
    return {"pump": spec["pump"], "tank": spec["tank"], "duration_ms": SPRAY_DURATION_MS}

@app.post("/predict")
async def predict(file: UploadFile = File(...), device_id: str = Form("browser"), api_key: str = Form("")):
    upsert_device(device_id)
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img_pil)
    if not detect_leaf(img_array):
        return {
            "disease": "No Leaf",
            "confidence": 0.0,
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": device_id,
            "message": "No leaf detected in image"
        }
    processed = preprocess_image(img_array)
    predictions = model.predict(processed)
    confidence = float(np.max(predictions))
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "disease": "Uncertain",
            "confidence": confidence,
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": device_id,
            "message": "Low confidence - point camera to a clear leaf"
        }
    disease = class_names[np.argmax(predictions)]
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("INSERT INTO results (timestamp, disease, confidence, device_id) VALUES (?,?,?,?)",
                (timestamp, disease, confidence, device_id))
    con.commit()
    con.close()

    # Only real ESP32 nodes may actuate pumps (browser scans never spray).
    action = None
    if device_id != "browser" and DEVICE_API_KEY and api_key == DEVICE_API_KEY:
        action = decide_action(device_id, disease)

    return {"disease": disease, "confidence": confidence, "timestamp": timestamp,
            "device_id": device_id, "action": action}

@app.get("/results")
def get_results(device_id: str = None):
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    if device_id:
        cur.execute("SELECT * FROM results WHERE device_id=? ORDER BY id DESC LIMIT 20", (device_id,))
    else:
        cur.execute("SELECT * FROM results ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    con.close()
    return {"data": rows}

@app.post("/pump/next")
def pump_next(device_id: str = Form(...), api_key: str = Form("")):
    """Pump controller polls this. Returns oldest fresh undelivered spray for device_id, marks delivered."""
    if not DEVICE_API_KEY or api_key != DEVICE_API_KEY:
        raise HTTPException(status_code=401, detail="bad api key")
    fresh = (datetime.datetime.utcnow() - timedelta(seconds=SPRAY_TTL_S)).strftime(TS_FMT)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, pump, duration_ms FROM sprays WHERE device_id=? AND delivered=0 "
                "AND timestamp>=? ORDER BY id LIMIT 1", (device_id, fresh))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE sprays SET delivered=1 WHERE id=?", (row[0],))
    cur.execute("UPDATE sprays SET delivered=2 WHERE delivered=0 AND timestamp<?", (fresh,))  # expired
    con.commit()
    con.close()
    if not row:
        return {"action": None}
    return {"action": {"pump": row[1], "duration_ms": row[2]}}

@app.get("/sprays")
def get_sprays():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT * FROM sprays ORDER BY id DESC LIMIT 50")
    rows = cur.fetchall()
    con.close()
    return {"data": rows}

@app.get("/devices")
def get_devices():
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("SELECT device_id, name, last_seen FROM devices ORDER BY last_seen DESC")
    rows = cur.fetchall()
    con.close()
    now = datetime.datetime.now()
    devices = []
    for r in rows:
        try:
            last = datetime.datetime.fromisoformat(r[2])
            secs = (now - last).total_seconds()
            online = secs < 90
        except:
            online = False
        devices.append({"device_id": r[0], "name": r[1], "last_seen": r[2], "online": online})
    return {"devices": devices}

@app.post("/devices/{device_id}/rename")
def rename_device(device_id: str, name: str = Form(...)):
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("UPDATE devices SET name=? WHERE device_id=?", (name, device_id))
    con.commit()
    con.close()
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()
