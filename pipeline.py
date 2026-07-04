"""
pipeline.py - Logika inti: cleaning, agregasi, feature engineering, GLM, XGBoost.
Dipakai bersama oleh app.py (versi teknis) dan app_toko.py (versi sederhana untuk toko).

CATATAN PENTING (perbaikan crash Windows):
statsmodels/scipy TIDAK diimpor di level atas file ini. Beberapa instalasi
Windows (kombinasi Python 3.13 + scipy + numpy tertentu) mengalami crash
fatal ("Fatal Python error: _PySemaphore_Wakeup") saat scipy diimpor lewat
statsmodels. Karena prediksi GLM sebenarnya hanya butuh operasi aljabar
sederhana (exp(X · koefisien) untuk Poisson GLM dengan log-link), jalur
PREDIKSI UTAMA (dipakai app_toko.py setiap kali dibuka) sengaja ditulis
ulang secara manual dengan numpy saja, TANPA statsmodels.

statsmodels/scipy hanya diimpor secara "lazy" (di dalam fungsi, bukan di
atas file) untuk fit_glm()/predict_glm() yang dipakai pada jalur RETRAIN
eksplisit (tab "Lanjutan" di app_toko.py) - jalur ini opsional dan jarang
dipakai, sehingga risikonya jauh lebih kecil dan tidak mengganggu
penggunaan aplikasi sehari-hari.
"""
import pandas as pd
import numpy as np
import os
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["lag_1", "lag_2", "lag_4", "roll_4", "roll_8", "week_num", "month", "quarter"]


def clean_data(df, col_map):
    df = df.rename(columns=col_map)
    before = len(df)
    df = df.drop_duplicates()
    df = df[df["jumlah_terjual"] > 0]
    df["jenis_pakan"] = df["jenis_pakan"].astype(str).str.strip().str.title()
    df["tanggal_transaksi"] = pd.to_datetime(df["tanggal_transaksi"])
    removed = before - len(df)
    return df, removed


def convert_to_kg(df, weight_col, already_kg):
    df = df.copy()
    if already_kg:
        df["jumlah_kg"] = df["jumlah_terjual"]
    else:
        df["jumlah_kg"] = df["jumlah_terjual"] * df[weight_col]
    return df


def aggregate_weekly(df):
    weekly = (
        df.set_index("tanggal_transaksi")
          .resample("W-MON")["jumlah_kg"]
          .sum()
          .reset_index()
          .rename(columns={"tanggal_transaksi": "minggu", "jumlah_kg": "total_kg"})
    )
    return weekly


def feature_engineering(weekly):
    df = weekly.copy().sort_values("minggu").reset_index(drop=True)
    df["lag_1"] = df["total_kg"].shift(1)
    df["lag_2"] = df["total_kg"].shift(2)
    df["lag_4"] = df["total_kg"].shift(4)
    df["roll_4"] = df["total_kg"].shift(1).rolling(4).mean()
    df["roll_8"] = df["total_kg"].shift(1).rolling(8).mean()
    df["week_num"] = df["minggu"].dt.isocalendar().week.astype(int)
    df["month"] = df["minggu"].dt.month
    df["quarter"] = df["minggu"].dt.quarter
    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df_feat, test_size=0.2):
    n = len(df_feat)
    n_test = max(1, int(n * test_size))
    return df_feat.iloc[: n - n_test], df_feat.iloc[n - n_test:]


# ============================================================
# GLM manual (numpy murni) - dipakai untuk jalur PREDIKSI UTAMA
# ============================================================
def predict_glm_manual(glm_coef, df):
    """Prediksi GLM Poisson (log-link) memakai koefisien murni (numpy),
    tanpa perlu statsmodels/scipy. Hasilnya identik dengan model.predict()
    versi statsmodels (sudah diverifikasi selisih = 0)."""
    param_names = glm_coef["param_names"]  # ['const', 'lag_1', ...]
    params = np.array(glm_coef["params"])
    X = np.column_stack([
        np.ones(len(df)) if name == "const" else df[name].values
        for name in param_names
    ])
    eta = X @ params
    return np.exp(eta)


# ============================================================
# GLM & XGBoost training (dipakai HANYA untuk retrain eksplisit)
# statsmodels diimpor "lazy" di sini, bukan di atas file.
# ============================================================
def fit_glm(train):
    import statsmodels.api as sm
    X_train = sm.add_constant(train[FEATURES])
    y_train = train["total_kg"]
    return sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()


def predict_glm(model, df):
    import statsmodels.api as sm
    X = sm.add_constant(df[FEATURES], has_constant="add")
    return model.predict(X)


def glm_model_to_coef(glm_model):
    """Ubah objek statsmodels GLM hasil fit_glm() menjadi koefisien murni
    (dict berisi param_names & params) yang bisa dipakai predict_glm_manual()
    tanpa statsmodels."""
    return {
        "param_names": list(glm_model.params.index),
        "params": glm_model.params.values.tolist(),
        "family": "poisson_log_link",
    }


def fit_xgb(train):
    model = XGBRegressor(objective="reg:squarederror", n_estimators=200,
                          learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(train[FEATURES], train["total_kg"])
    return model


def evaluate(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    return mae, rmse, mape


def recursive_forecast(model_or_coef, df_feat, n_weeks, model_type):
    """model_or_coef: untuk GLM, ini adalah dict koefisien (dari glm_coef.pkl).
    Untuk XGBoost, ini adalah objek XGBRegressor seperti biasa."""
    history = df_feat["total_kg"].tolist()
    last_date = df_feat["minggu"].max()
    preds = []
    for i in range(1, n_weeks + 1):
        next_date = last_date + pd.Timedelta(weeks=i)
        lag_1 = history[-1]
        lag_2 = history[-2] if len(history) >= 2 else lag_1
        lag_4 = history[-4] if len(history) >= 4 else lag_1
        roll_4 = np.mean(history[-4:]) if len(history) >= 4 else np.mean(history)
        roll_8 = np.mean(history[-8:]) if len(history) >= 8 else np.mean(history)
        feat = pd.DataFrame([{
            "lag_1": lag_1, "lag_2": lag_2, "lag_4": lag_4,
            "roll_4": roll_4, "roll_8": roll_8,
            "week_num": int(next_date.isocalendar().week),
            "month": next_date.month, "quarter": next_date.quarter,
        }])
        if model_type == "GLM":
            pred = predict_glm_manual(model_or_coef, feat)[0]
        else:
            pred = model_or_coef.predict(feat[FEATURES])[0]
        pred = max(0, pred)
        preds.append({"minggu": next_date.date(), "prediksi_kg": round(pred, 1)})
        history.append(pred)
    return pd.DataFrame(preds)


def run_full_pipeline(df_raw, col_map, weight_col, already_kg, test_pct=0.2):
    """Jalankan seluruh pipeline sekali panggil - dipakai untuk RETRAIN eksplisit
    (bukan default), misalnya saat data historis bertambah signifikan.
    CATATAN: fungsi ini butuh statsmodels/scipy (diimpor lazy di fit_glm/predict_glm).
    Kalau lingkungan bermasalah dengan scipy, fungsi ini bisa gagal - itulah
    kenapa retrain dipisah sebagai fitur opsional, bukan jalur utama aplikasi."""
    df_clean, removed = clean_data(df_raw, col_map)
    df_kg = convert_to_kg(df_clean, weight_col, already_kg)
    weekly = aggregate_weekly(df_kg)
    df_feat = feature_engineering(weekly)

    if len(df_feat) < 15:
        raise ValueError("Data mingguan terlalu sedikit (minimal ~15 minggu histori) untuk membuat prediksi.")

    train, test = split_data(df_feat, test_size=test_pct)
    glm_model = fit_glm(train)
    xgb_model = fit_xgb(train)
    glm_pred = predict_glm(glm_model, test)
    xgb_pred = xgb_model.predict(test[FEATURES])

    mae_glm, rmse_glm, mape_glm = evaluate(test["total_kg"], glm_pred)
    mae_xgb, rmse_xgb, mape_xgb = evaluate(test["total_kg"], xgb_pred)
    best_name = "GLM" if mae_glm <= mae_xgb else "XGBoost"
    best_model = glm_model if best_name == "GLM" else xgb_model

    return {
        "df_clean": df_clean, "weekly": weekly, "df_feat": df_feat,
        "train": train, "test": test,
        "glm_model": glm_model, "xgb_model": xgb_model,
        "glm_pred": glm_pred, "xgb_pred": xgb_pred,
        "metrics": {
            "GLM": {"MAE": mae_glm, "RMSE": rmse_glm, "MAPE": mape_glm},
            "XGBoost": {"MAE": mae_xgb, "RMSE": rmse_xgb, "MAPE": mape_xgb},
        },
        "best_model_name": best_name, "best_model": best_model,
        "removed": removed,
    }


def load_final_model(model_dir="model_final"):
    """
    Memuat model FINAL yang sudah dilatih sekali dari data asli dan dikunci
    (random_state=42), sesuai rancangan Bab IV.

    PENTING: untuk GLM, yang dimuat adalah KOEFISIEN MURNI (glm_coef.pkl),
    bukan objek statsmodels - supaya tidak butuh statsmodels/scipy sama
    sekali di jalur prediksi utama (menghindari crash di beberapa instalasi
    Windows). xgb_model.pkl tetap dimuat sebagai objek XGBRegressor biasa
    (XGBoost tidak bermasalah dengan crash ini).
    """
    with open(os.path.join(model_dir, "glm_coef.pkl"), "rb") as f:
        glm_coef = pickle.load(f)
    with open(os.path.join(model_dir, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(model_dir, "metrics.pkl"), "rb") as f:
        metrics = pickle.load(f)
    df_feat = pd.read_csv(os.path.join(model_dir, "df_feat.csv"), parse_dates=["minggu"])

    best_name = metrics["best_model"]
    best_model = glm_coef if best_name == "GLM" else xgb_model

    return {
        "df_feat": df_feat,
        "glm_coef": glm_coef, "xgb_model": xgb_model,
        "metrics": {"GLM": metrics["GLM"], "XGBoost": metrics["XGBoost"]},
        "best_model_name": best_name, "best_model": best_model,
    }
