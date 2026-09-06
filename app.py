import os
import json
import re
import hmac
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from training_service import TrainingService

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)


MODEL_PATH = MODEL_DIR / "model_lightgbm_umkm.pkl"
METADATA_PATH = MODEL_DIR / "metadata_umkm.json"
MAPPING_PATH = MODEL_DIR / "mapping_usaha_umkm.json"
METRICS_PATH = MODEL_DIR / "metrics_umkm.json"

model = None
metadata = {}
mapping_data = []
metrics = {}

FEATURE_COLS = [
    "Modal",
    "Potensi_Pasar",
    "Lokasi",
    "Persaingan",
    "Tren_Usaha"
]

CATEGORY_MAP = {
    0: "Kuliner",
    1: "Perdagangan",
    2: "Jasa",
    3: "Produksi",
    4: "Pertanian/Peternakan"
}

INPUT_MAPS = {
    "Potensi_Pasar": {
        "1": "Rendah",
        "2": "Sedang",
        "3": "Tinggi"
    },
    "Lokasi": {
        "1": "Kurang Strategis",
        "2": "Strategis",
        "3": "Sangat Strategis"
    },
    "Persaingan": {
        "1": "Rendah",
        "2": "Sedang",
        "3": "Tinggi"
    },
    "Tren_Usaha": {
        "1": "Kurang Stabil",
        "2": "Stabil",
        "3": "Sangat Stabil"
    }
}


training_service = TrainingService(BASE_DIR, MODEL_DIR)


def load_artifacts():
    global model, metadata, mapping_data, metrics, FEATURE_COLS, CATEGORY_MAP, INPUT_MAPS

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        FEATURE_COLS = metadata.get("feature_cols", FEATURE_COLS)

        if "category_map" in metadata:
            CATEGORY_MAP = {
                int(k): v for k, v in metadata["category_map"].items()
            }

        if "input_maps" in metadata:
            INPUT_MAPS = metadata["input_maps"]

    if MAPPING_PATH.exists():
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)


def get_mapping_usaha(kategori, top_n=3):
    hasil = []

    for item in mapping_data:
        if item.get("Kategori") == kategori:
            hasil.append(item.get("Jenis_Usaha"))

    hasil = [x for x in hasil if x]

    return hasil[:top_n]


def status_confidence(confidence):
    if confidence >= 0.80:
        return "Sangat yakin"
    if confidence >= 0.60:
        return "Cukup yakin"
    return "Kurang yakin"


def buat_catatan(modal, tren_usaha, confidence):
    catatan = []

    if modal < 500000:
        catatan.append(
            "Modal yang dimasukkan tergolong sangat rendah, sehingga usaha sebaiknya dimulai dari skala sangat kecil atau rumahan."
        )
    elif modal < 1000000:
        catatan.append(
            "Modal tergolong rendah, sehingga usaha yang dipilih sebaiknya dimulai dari skala kecil."
        )

    if tren_usaha == 1:
        catatan.append(
            "Kondisi tren usaha tergolong kurang stabil, sehingga perlu kehati-hatian dalam memilih produk, harga, dan strategi pemasaran."
        )

    if confidence < 0.60:
        catatan.append(
            "Tingkat keyakinan model belum kuat, sehingga pengguna perlu membandingkan beberapa alternatif kategori usaha."
        )

    return catatan


def alasan_fallback(hasil_model):
    hasil = hasil_model["hasil_rekomendasi"]
    kategori = hasil["kategori_rekomendasi"]
    confidence = hasil["confidence_persen"]
    detail = hasil["rekomendasi_detail_usaha"]
    catatan = hasil.get("catatan", [])

    detail_text = ", ".join(detail) if detail else "belum tersedia"
    catatan_text = " ".join(catatan) if catatan else "Tidak ada catatan khusus dari sistem."

    return (
        f"Berdasarkan hasil perhitungan model LightGBM, sistem merekomendasikan kategori "
        f"{kategori} dengan tingkat keyakinan {confidence}%. "
        f"Rekomendasi usaha yang dapat dipertimbangkan adalah {detail_text}. "
        f"{catatan_text} "
        f"Hasil ini bersifat rekomendasi awal, sehingga pengguna tetap perlu mempertimbangkan kondisi lapangan."
    )


def buat_alasan_ai(hasil_model):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_ai = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    if not api_key or OpenAI is None:
        return alasan_fallback(hasil_model)

    try:
        client = OpenAI(api_key=api_key)

        prompt_data = json.dumps(hasil_model, ensure_ascii=False, indent=2)

        prompt = f"""
Anda adalah asisten Sistem Pendukung Keputusan untuk rekomendasi jenis usaha UMKM.

Data berikut adalah hasil dari model LightGBM.
Anda tidak boleh mengubah hasil prediksi model.
Anda tidak boleh mengganti kategori rekomendasi.
Anda tidak boleh menambah jenis usaha di luar daftar mapping.
Tugas Anda hanya menjelaskan alasan hasil tersebut.

Data hasil model:
{prompt_data}

Buat jawaban dalam bahasa Indonesia dengan format:

Analisis Input:
...

Alasan Rekomendasi:
...

Penjelasan Mapping Usaha:
...

Catatan Perhatian:
...

Kesimpulan:
...
"""

        response = client.responses.create(
            model=model_ai,
            input=prompt,
            max_output_tokens=600
        )

        return response.output_text

    except Exception:
        return alasan_fallback(hasil_model)


def validasi_payload(data):
    required = ["modal", "potensi_pasar", "lokasi", "persaingan", "tren_usaha"]

    for key in required:
        if key not in data:
            return False, f"Field {key} wajib dikirim."

    try:
        modal = int(data["modal"])
        potensi_pasar = int(data["potensi_pasar"])
        lokasi = int(data["lokasi"])
        persaingan = int(data["persaingan"])
        tren_usaha = int(data["tren_usaha"])
    except Exception:
        return False, "Semua input harus berupa angka."

    if modal <= 0:
        return False, "Modal harus lebih dari 0."

    for key, value in {
        "potensi_pasar": potensi_pasar,
        "lokasi": lokasi,
        "persaingan": persaingan,
        "tren_usaha": tren_usaha,
    }.items():
        if value not in [1, 2, 3]:
            return False, f"Nilai {key} harus 1, 2, atau 3."

    return True, "Valid"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": True,
        "message": "Flask API LightGBM aktif",
        "model_ready": model is not None,
        "model_path": str(MODEL_PATH),
    })


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify({
        "status": True,
        "message": "Metrics model",
        "data": metrics
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status": False,
            "message": "Model belum tersedia. Pastikan file model_lightgbm_umkm.pkl ada di folder flask_api/model."
        }), 500

    data = request.get_json(silent=True) or {}

    valid, message = validasi_payload(data)

    if not valid:
        return jsonify({
            "status": False,
            "message": message
        }), 400

    modal = int(data["modal"])
    potensi_pasar = int(data["potensi_pasar"])
    lokasi = int(data["lokasi"])
    persaingan = int(data["persaingan"])
    tren_usaha = int(data["tren_usaha"])

    top_n = int(data.get("top_n", 3))
    top_n = max(1, min(top_n, 3))

    input_df = pd.DataFrame([{
        "Modal": modal,
        "Potensi_Pasar": potensi_pasar,
        "Lokasi": lokasi,
        "Persaingan": persaingan,
        "Tren_Usaha": tren_usaha,
    }], columns=FEATURE_COLS)

    proba = model.predict_proba(input_df)[0]
    pred_label = int(np.argmax(proba))
    pred_category = CATEGORY_MAP[pred_label]
    confidence = float(proba[pred_label])

    detail_usaha = get_mapping_usaha(pred_category, top_n=top_n)

    probabilitas_semua = {
        CATEGORY_MAP[int(i)]: round(float(p) * 100, 2)
        for i, p in enumerate(proba)
    }

    hasil_model = {
        "input": {
            "Modal": modal,
            "Potensi_Pasar": potensi_pasar,
            "Lokasi": lokasi,
            "Persaingan": persaingan,
            "Tren_Usaha": tren_usaha,
        },
        "hasil_rekomendasi": {
            "label_prediksi": pred_label,
            "kategori_rekomendasi": pred_category,
            "confidence_persen": round(confidence * 100, 2),
            "status_confidence": status_confidence(confidence),
            "rekomendasi_detail_usaha": detail_usaha,
            "probabilitas_semua_kategori_persen": probabilitas_semua,
            "catatan": buat_catatan(modal, tren_usaha, confidence),
        }
    }

    if bool(data.get("include_ai_explanation", False)):
        hasil_model["hasil_rekomendasi"]["alasan_ai"] = buat_alasan_ai(hasil_model)
    else:
        hasil_model["hasil_rekomendasi"]["alasan_ai"] = None

    return jsonify({
        "status": True,
        "message": "Prediksi berhasil",
        "data": hasil_model
    })


def training_authorized():
    """
    Jika TRAINING_API_TOKEN diisi di .env Flask, endpoint training wajib
    menerima header X-Training-Token dengan nilai yang sama. Jika token
    dibiarkan kosong, endpoint tetap dapat dipakai untuk lingkungan lokal.
    """
    configured = os.getenv("TRAINING_API_TOKEN", "").strip()
    if not configured:
        return True

    supplied = request.headers.get("X-Training-Token", "").strip()
    return bool(supplied) and hmac.compare_digest(configured, supplied)


def training_unauthorized_response():
    return jsonify({
        "status": False,
        "message": "Akses endpoint training ditolak. Token training tidak valid."
    }), 401


@app.route("/training/validate", methods=["POST"])
def training_validate():
    if not training_authorized():
        return training_unauthorized_response()

    uploaded = request.files.get("dataset")
    if uploaded is None or not uploaded.filename:
        return jsonify({
            "status": False,
            "message": "File dataset CSV wajib dikirim pada field 'dataset'."
        }), 400

    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify({
            "status": False,
            "message": "Dataset wajib menggunakan format .csv."
        }), 400

    try:
        df = training_service.read_csv(uploaded.stream)
        _, summary = training_service.validate_dataframe(df)
        return jsonify({
            "status": True,
            "message": "Dataset valid dan siap digunakan untuk training.",
            "data": summary
        })
    except Exception as exc:
        return jsonify({
            "status": False,
            "message": str(exc)
        }), 400


@app.route("/training/train", methods=["POST"])
def training_train():
    if not training_authorized():
        return training_unauthorized_response()

    uploaded = request.files.get("dataset")
    if uploaded is None or not uploaded.filename:
        return jsonify({
            "status": False,
            "message": "File dataset CSV wajib dikirim pada field 'dataset'."
        }), 400

    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify({
            "status": False,
            "message": "Dataset wajib menggunakan format .csv."
        }), 400

    notes = (request.form.get("notes") or "").strip()

    try:
        df = training_service.read_csv(uploaded.stream)
        result = training_service.train_candidate(
            df=df,
            dataset_name=uploaded.filename,
            notes=notes,
        )
        return jsonify({
            "status": True,
            "message": result["message"],
            "data": result
        })
    except Exception as exc:
        return jsonify({
            "status": False,
            "message": str(exc)
        }), 400


@app.route("/training/models", methods=["GET"])
def training_models():
    if not training_authorized():
        return training_unauthorized_response()

    try:
        return jsonify({
            "status": True,
            "message": "Daftar model berhasil dimuat.",
            "data": training_service.list_models()
        })
    except Exception as exc:
        return jsonify({
            "status": False,
            "message": str(exc)
        }), 500


@app.route("/training/activate/<model_id>", methods=["POST"])
def training_activate(model_id):
    if not training_authorized():
        return training_unauthorized_response()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id or ""):
        return jsonify({
            "status": False,
            "message": "ID model tidak valid."
        }), 400

    try:
        result = training_service.activate_candidate(model_id)
        # Reload model dan metadata tanpa perlu restart Flask.
        load_artifacts()
        return jsonify({
            "status": True,
            "message": result["message"],
            "data": result
        })
    except Exception as exc:
        return jsonify({
            "status": False,
            "message": str(exc)
        }), 400


load_artifacts()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))

    app.run(host=host, port=port, debug=True)