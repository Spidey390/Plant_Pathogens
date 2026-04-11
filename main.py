from fastapi import FastAPI, File, UploadFile
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
        confidence REAL
    )""")
    con.commit()
    con.close()

init_db()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
    cur.execute("INSERT INTO results (timestamp, disease, confidence) VALUES (?,?,?)", (timestamp, disease, confidence))
    con.commit()
    con.close()
    return {"disease": disease, "confidence": confidence, "timestamp": timestamp}

@app.get("/results")
def get_results():
    con = sqlite3.connect("results.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM results ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    con.close()
    return {"data": rows}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("templates/index.html") as f:
        return f.read()