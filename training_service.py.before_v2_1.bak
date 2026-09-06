import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "Modal",
    "Potensi_Pasar",
    "Lokasi",
    "Persaingan",
    "Tren_Usaha",
]
TARGET_COL = "Jenis_Usaha"

CATEGORY_MAP = {
    0: "Kuliner",
    1: "Perdagangan",
    2: "Jasa",
    3: "Produksi",
    4: "Pertanian/Peternakan",
}
CATEGORY_TO_ID = {v: k for k, v in CATEGORY_MAP.items()}

CATEGORY_ALIASES = {
    "Produksi Rumah Tangga": "Produksi",
    "Pertanian/Peternakan Skala Kecil": "Pertanian/Peternakan",
    "Pertanian / Peternakan": "Pertanian/Peternakan",
}

INPUT_MAPS = {
    "Potensi_Pasar": {"1": "Rendah", "2": "Sedang", "3": "Tinggi"},
    "Lokasi": {"1": "Kurang Strategis", "2": "Strategis", "3": "Sangat Strategis"},
    "Persaingan": {"1": "Rendah", "2": "Sedang", "3": "Tinggi"},
    "Tren_Usaha": {"1": "Kurang Stabil", "2": "Stabil", "3": "Sangat Stabil"},
}

MODEL_PARAMS = {
    "objective": "multiclass",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1,
}


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TrainingService:
    def __init__(self, base_dir: Path, model_dir: Path):
        self.base_dir = Path(base_dir)
        self.model_dir = Path(model_dir)
        self.training_dir = self.base_dir / "training"
        self.dataset_dir = self.training_dir / "datasets"
        self.candidate_dir = self.training_dir / "candidates"
        self.archive_dir = self.training_dir / "archive"
        self.history_path = self.training_dir / "history.json"

        for directory in [
            self.training_dir,
            self.dataset_dir,
            self.candidate_dir,
            self.archive_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        if not self.history_path.exists():
            self._write_json(self.history_path, [])

    @staticmethod
    def _write_json(path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    @staticmethod
    def _read_json(path: Path, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _history(self):
        data = self._read_json(self.history_path, [])
        if not isinstance(data, list):
            return []

        # Kompatibilitas dengan history lama/korup yang mungkin berisi string.
        # Endpoint /training/models dan aktivasi model hanya memproses entry dict.
        return [item for item in data if isinstance(item, dict)]

    def _save_history(self, history):
        self._write_json(self.history_path, history)

    @staticmethod
    def read_csv(file_obj):
        """Membaca CSV upload dengan toleransi UTF-8 BOM."""
        try:
            return pd.read_csv(file_obj, encoding="utf-8-sig")
        except UnicodeDecodeError:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            return pd.read_csv(file_obj, encoding="latin-1")

    def validate_dataframe(self, df: pd.DataFrame):
        required = FEATURE_COLS + [TARGET_COL]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing))

        df = df[required].copy()

        if len(df) < 100:
            raise ValueError("Dataset minimal berisi 100 baris agar pembagian training/validation/testing lebih stabil.")

        for col in FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[FEATURE_COLS].isna().any().any():
            bad_cols = df[FEATURE_COLS].columns[df[FEATURE_COLS].isna().any()].tolist()
            raise ValueError("Terdapat nilai kosong/non-numerik pada kolom: " + ", ".join(bad_cols))

        df["Modal"] = df["Modal"].round().astype("int64")
        if (df["Modal"] <= 0).any():
            raise ValueError("Semua nilai Modal harus lebih dari 0.")

        for col in ["Potensi_Pasar", "Lokasi", "Persaingan", "Tren_Usaha"]:
            df[col] = df[col].round().astype("int64")
            invalid = sorted(set(df.loc[~df[col].isin([1, 2, 3]), col].tolist()))
            if invalid:
                raise ValueError(f"Kolom {col} hanya boleh berisi nilai 1, 2, atau 3. Nilai tidak valid: {invalid[:10]}")

        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().replace(CATEGORY_ALIASES)

        unknown = sorted(set(df[TARGET_COL]) - set(CATEGORY_TO_ID))
        if unknown:
            raise ValueError(
                "Kategori target tidak dikenali: " + ", ".join(unknown)
                + ". Gunakan: " + ", ".join(CATEGORY_TO_ID.keys())
            )

        present = set(df[TARGET_COL])
        missing_classes = [c for c in CATEGORY_TO_ID if c not in present]
        if missing_classes:
            raise ValueError("Dataset harus memuat kelima kategori. Kategori yang belum ada: " + ", ".join(missing_classes))

        distribution = df[TARGET_COL].value_counts().reindex(CATEGORY_TO_ID.keys(), fill_value=0).to_dict()
        too_small = {k: int(v) for k, v in distribution.items() if int(v) < 20}
        if too_small:
            raise ValueError(
                "Setiap kategori minimal memiliki 20 data agar stratified split dapat dilakukan dengan aman. "
                f"Kategori kurang: {too_small}"
            )

        duplicate_count = int(df.duplicated().sum())

        modal_by_category = {}
        for kategori, group in df.groupby(TARGET_COL):
            modal_by_category[str(kategori)] = {
                "count": int(len(group)),
                "min": int(group["Modal"].min()),
                "max": int(group["Modal"].max()),
                "median": int(group["Modal"].median()),
                "mean": round(float(group["Modal"].mean()), 2),
            }

        summary = {
            "total_rows": int(len(df)),
            "columns": required,
            "distribution": {k: int(v) for k, v in distribution.items()},
            "duplicate_rows": duplicate_count,
            "modal_min": int(df["Modal"].min()),
            "modal_max": int(df["Modal"].max()),
            "modal_by_category": modal_by_category,
            "criteria_values": {
                col: sorted(int(x) for x in df[col].unique().tolist())
                for col in ["Potensi_Pasar", "Lokasi", "Persaingan", "Tren_Usaha"]
            },
        }

        return df, summary

    def train_candidate(self, df: pd.DataFrame, dataset_name: str, notes: str = ""):
        df, summary = self.validate_dataframe(df)

        X = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].map(CATEGORY_TO_ID).astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=42,
        )

        candidate_model = LGBMClassifier(**MODEL_PARAMS)
        candidate_model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
        )

        pred = candidate_model.predict(X_test)
        labels = list(CATEGORY_MAP.keys())
        target_names = [CATEGORY_MAP[i] for i in labels]

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 6),
            "precision_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 6),
            "recall_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 6),
            "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 6),
            "classification_report": classification_report(
                y_test,
                pred,
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        }

        feature_importance = {
            feature: int(value)
            for feature, value in zip(FEATURE_COLS, candidate_model.feature_importances_)
        }
        metrics["feature_importance"] = feature_importance

        model_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        model_path = self.candidate_dir / model_id
        model_path.mkdir(parents=True, exist_ok=False)

        safe_dataset_name = Path(dataset_name or "dataset.csv").name
        saved_dataset_name = f"{model_id}_{safe_dataset_name}"
        dataset_save_path = self.dataset_dir / saved_dataset_name
        df.to_csv(dataset_save_path, index=False, encoding="utf-8-sig")

        metadata = {
            "model_id": model_id,
            "created_at": utc_now_iso(),
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
            "category_map": {str(k): v for k, v in CATEGORY_MAP.items()},
            "input_maps": INPUT_MAPS,
            "model_params": MODEL_PARAMS,
            "split": {
                "training": int(len(X_train)),
                "validation": int(len(X_valid)),
                "testing": int(len(X_test)),
                "training_percent": 80,
                "validation_percent": 10,
                "testing_percent": 10,
                "method": "Stratified Sampling",
                "random_state": 42,
            },
            "dataset": {
                "name": safe_dataset_name,
                "saved_name": saved_dataset_name,
                **summary,
            },
            "notes": notes[:500],
        }

        joblib.dump(candidate_model, model_path / "model_lightgbm_umkm.pkl")
        self._write_json(model_path / "metadata_umkm.json", metadata)
        self._write_json(model_path / "metrics_umkm.json", metrics)

        pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in feature_importance.items()]
        ).to_csv(model_path / "feature_importance_umkm.csv", index=False)

        history = self._history()
        entry = {
            "model_id": model_id,
            "status": "candidate",
            "created_at": metadata["created_at"],
            "dataset_name": safe_dataset_name,
            "total_rows": summary["total_rows"],
            "distribution": summary["distribution"],
            "split": metadata["split"],
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            },
            "notes": notes[:500],
        }
        history.insert(0, entry)
        self._save_history(history)

        return {
            "model_id": model_id,
            "status": "candidate",
            "dataset": summary,
            "split": metadata["split"],
            "metrics": metrics,
            "message": "Model berhasil dilatih dan disimpan sebagai kandidat. Model aktif belum berubah.",
        }

    def _active_from_files(self):
        metadata_path = self.model_dir / "metadata_umkm.json"
        metrics_path = self.model_dir / "metrics_umkm.json"
        model_path = self.model_dir / "model_lightgbm_umkm.pkl"

        if not model_path.exists():
            return None

        metadata = self._read_json(metadata_path, {})
        metrics = self._read_json(metrics_path, {})

        # Metadata model lama dapat menyimpan field `dataset` sebagai string,
        # sedangkan model baru menyimpannya sebagai object/dict. Normalisasi agar
        # endpoint /training/models tetap kompatibel dengan keduanya.
        dataset_name = "model legacy"
        if isinstance(metadata, dict):
            raw_dataset = metadata.get("dataset")
            if isinstance(raw_dataset, dict):
                dataset_name = raw_dataset.get("name") or metadata.get("dataset_name") or "model legacy"
            elif isinstance(raw_dataset, str) and raw_dataset.strip():
                dataset_name = raw_dataset.strip()
            else:
                dataset_name = metadata.get("dataset_name") or "model legacy"

        return {
            "model_id": metadata.get("model_id", "model_aktif_saat_ini") if isinstance(metadata, dict) else "model_aktif_saat_ini",
            "status": "active",
            "created_at": metadata.get("created_at") if isinstance(metadata, dict) else None,
            "dataset_name": dataset_name,
            "metrics": {
                "accuracy": metrics.get("accuracy"),
                "precision_macro": metrics.get("precision_macro"),
                "recall_macro": metrics.get("recall_macro"),
                "f1_macro": metrics.get("f1_macro"),
            } if isinstance(metrics, dict) else {},
        }

    def list_models(self):
        return {
            "active_model": self._active_from_files(),
            "models": self._history(),
        }

    def activate_candidate(self, model_id: str):
        candidate = self.candidate_dir / model_id
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError("Model kandidat tidak ditemukan.")

        required = [
            "model_lightgbm_umkm.pkl",
            "metadata_umkm.json",
            "metrics_umkm.json",
        ]
        missing = [name for name in required if not (candidate / name).exists()]
        if missing:
            raise ValueError("Artifact kandidat tidak lengkap: " + ", ".join(missing))

        self.model_dir.mkdir(parents=True, exist_ok=True)

        archive_name = "before_" + model_id + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / archive_name
        archive_path.mkdir(parents=True, exist_ok=False)

        for name in [
            "model_lightgbm_umkm.pkl",
            "metadata_umkm.json",
            "metrics_umkm.json",
            "feature_importance_umkm.csv",
        ]:
            src = self.model_dir / name
            if src.exists():
                shutil.copy2(src, archive_path / name)

        for name in [
            "model_lightgbm_umkm.pkl",
            "metadata_umkm.json",
            "metrics_umkm.json",
            "feature_importance_umkm.csv",
        ]:
            src = candidate / name
            if src.exists():
                shutil.copy2(src, self.model_dir / name)

        history = self._history()
        found = False
        for item in history:
            if item.get("status") == "active":
                item["status"] = "archived"
            if item.get("model_id") == model_id:
                item["status"] = "active"
                item["activated_at"] = utc_now_iso()
                found = True

        if not found:
            raise ValueError("Riwayat model kandidat tidak ditemukan.")

        self._save_history(history)

        return {
            "model_id": model_id,
            "status": "active",
            "archive": archive_name,
            "message": "Model kandidat berhasil diaktifkan. Model sebelumnya telah diarsipkan.",
        }
