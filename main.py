from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import sqlite3
import datetime
import io
import os
import gdown
from PIL import Image
MODEL_PATH = "plant_disease_model.h5"
GDRIVE_FILE_ID = "13e9PTZCsu10q8l0tnRSmQjjL8Kh5h9pK"

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
    con.commit()
    con.close()

init_db()

def upsert_device(device_id: str):
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("SELECT device_id FROM devices WHERE device_id=?", (device_id,))
    exists = cur.fetchone()
    now = datetime.datetime.now().isoformat()
    if exists:
        cur.execute("UPDATE devices SET last_seen=? WHERE device_id=?", (now, device_id))
    else:
        cur.execute("INSERT INTO devices (device_id, name, last_seen) VALUES (?,?,?)", (device_id, device_id, now))
    con.commit()
    con.close()

@app.post("/predict")
async def predict(file: UploadFile = File(...), device_id: str = Form("browser")):
    upsert_device(device_id)
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    disease = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    timestamp = datetime.datetime.now().isoformat()
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("INSERT INTO results (timestamp, disease, confidence, device_id) VALUES (?,?,?,?)",
                (timestamp, disease, confidence, device_id))
    con.commit()
    con.close()
    return {"disease": disease, "confidence": confidence, "timestamp": timestamp, "device_id": device_id}

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
    with open("templates/index.html") as f:
        return f.read()