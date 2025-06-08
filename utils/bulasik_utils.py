#bulaşık utils 
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------- AYARLAR --------------------
DATA_PATH = r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/guncel_Bulasık_islenmis_onehot.xlsx"
MODEL_PATH = r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/mühp2/models/bulasik_model.pkl"

# -------------------- 1. VERİ YÜKLEME --------------------
def load_and_process_data(path):
    print("📥 Veri yükleniyor ve işleniyor...")
    df = pd.read_excel(path)

    df["datetime"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["datetime"].dt.date

    daily_consumption = df.groupby("date")["power_watt"].sum()
    threshold = daily_consumption.mean()

    df["Inefficient"] = df["date"].map(lambda d: int(daily_consumption[d] > threshold))

    return df, threshold

# -------------------- 2. MODEL EĞİTİMİ --------------------
def train_model(df, model_path):
    print("🔧 Model (GridSearch ile) eğitiliyor...")

    feature_cols = [col for col in df.columns if col not in ["timestamp", "datetime", "date", "Inefficient"]]
    X = df[feature_cols]
    y = df["Inefficient"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],  # l1 destekliyor
        "class_weight": ["balanced"]
    }

    grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                        param_grid,
                        cv=5,
                        scoring="f1",
                        n_jobs=-1)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("🔎 En iyi parametreler:", grid.best_params_)

    print("Model Değerlendirme:")
    print(classification_report(y_test, best_model.predict(X_test)))

    cv = cross_val_score(best_model, X, y, cv=5, scoring="f1")
    print(f"5-fold CV F1 Skoru: {cv.mean():.3f} ± {cv.std():.3f}")

    joblib.dump(best_model, model_path)
    print(f"✅ Model '{model_path}' olarak kaydedildi.")
    return best_model

# -------------------- 3. EN SON GÜNÜ GETİR --------------------
def get_last_day_data(df):
    last_date = df["date"].max()
    df_last = df[df["date"] == last_date]
    print(f"Son gün verisi: {last_date}, kayıt sayısı: {len(df_last)}")
    return df_last

# -------------------- 4. VERİMLİLİK TAHMİNİ --------------------
def predict_efficiency(df_day, model_path, threshold_prob=0.7, threshold_daily_factor=1.5, threshold_kwh=None):
    print("🤖 Model ile verimlilik tahmini yapılıyor...")

    feature_cols = [col for col in df_day.columns if col not in ["timestamp", "datetime", "date", "Inefficient"]]
    X_avg = df_day[feature_cols].mean().to_frame().T

    model = joblib.load(model_path)
    proba = model.predict_proba(X_avg)[0][1]
    print(f"Inefficient probability: {proba:.3f}")

    daily_kwh = df_day["power_watt"].sum()

    if proba > threshold_prob or (threshold_kwh is not None and daily_kwh > threshold_kwh * threshold_daily_factor):
        status = "verimsiz"
    else:
        status = "verimli"

    return status, proba, daily_kwh

# -------------------- 5. ÖNERİ OLUŞTUR --------------------
def generate_advice(status, prob, daily_kwh, threshold, df_day):
    print(f"\n📊 Günlük tüketim: {daily_kwh:.2f} Wh")
    print(f"🎯 Verimlilik durumu: {status.capitalize()}")

    if status == "verimsiz":
        if prob >= 0.8:
            msg = "🚨 Bulaşık makinesi çok sık kullanılıyor. Tam dolmadan çalıştırılmamalı ve kısa programlar tercih edilmeli."
        elif "is_peak" in df_day.columns and df_day["is_peak"].sum() > len(df_day) * 0.3:
            msg = "⏱️ Puant saatlerde yıkama yapılmış. Elektrik maliyetini azaltmak için gece saatleri daha uygun olabilir."
        else:
            msg = "⚠️ Ortalama tüketim yüksek. Eko mod tercih edilerek enerji tasarrufu sağlanabilir."
    else:
        if df_day["power_watt"].max() > threshold * 0.1:
            msg = "ℹ️ Gün içinde kısa süreli yüksek tüketim olmuş. Bu durum yoğun program seçiminden kaynaklanabilir."
        else:
            msg = "✅ Bulaşık makinesi verimli kullanılıyor. Mevcut kullanım alışkanlıklarınız uygundur."

    print(f"💡 Öneri: {msg}")
    return msg

# -------------------- ANA AKIŞ --------------------
if __name__ == "__main__":
    df_all, threshold = load_and_process_data(DATA_PATH)
    model = train_model(df_all, MODEL_PATH)

    df_last = get_last_day_data(df_all)
    status, prob, daily_kwh = predict_efficiency(df_last, MODEL_PATH, threshold_prob=0.5, threshold_kwh=threshold)
    advice = generate_advice(status, prob, daily_kwh, threshold, df_last)
