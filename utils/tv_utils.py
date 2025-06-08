import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------- AYARLAR --------------------
DATA_PATH = r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/TELEVIZYON_islenmis_onehot.xlsx"
MODEL_PATH = r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/mühp2/models/tv_model.pkl"

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
    print("🔧 Model (regularize edilmiş) eğitiliyor...")

    feature_cols = [col for col in df.columns if col not in ["timestamp", "datetime", "date", "Inefficient"]]
    X = df[feature_cols]
    y = df["Inefficient"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.001,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Model Değerlendirme:")
    print(classification_report(y_test, model.predict(X_test)))

    cv = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"5-fold CV F1 Skoru: {cv.mean():.3f} ± {cv.std():.3f}")

    joblib.dump(model, model_path)
    print(f"✅ Model '{model_path}' olarak kaydedildi.")
    return model

# -------------------- 3. EN SON GÜNÜ GETİR --------------------
def get_last_day_data(df):
    last_date = df["date"].max()
    df_last = df[df["date"] == last_date]
    print(f"Son gün verisi: {last_date}, kayıt sayısı: {len(df_last)}")
    return df_last

# -------------------- 4. VERİMLİLİK TAHMİNİ --------------------
def predict_efficiency(df_day, model_path, threshold_prob=0.1, threshold_daily_factor=1.5, threshold_kwh=None):
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
            msg = "🚨 TV çok uzun süre açık kalıyor. Kullanım süresini azaltmayı düşünün."
        elif "is_peak" in df_day.columns and df_day["is_peak"].sum() > len(df_day) * 0.3:
            msg = "⏱️ TV puant saatlerde sık kullanılıyor. Daha düşük tarifeli saatlerde izlemeye çalışın."
        else:
            msg = "⚠️ Ortalama tüketim yüksek. TV'nin gereksiz açık bırakılmadığından emin olun."
    else:
        if df_day["power_watt"].max() > threshold * 0.1:
            msg = "⚠️ Bazı saatlerde TV'nin tüketimi çok yüksek, arka planda açık kalmış olabilir."
        else:
            msg = "✅ Televizyon verimli çalışıyor. Mevcut kullanım şekli uygundur."

    print(f"💡 Öneri: {msg}")
    return msg

# -------------------- ANA AKIŞ --------------------
if __name__ == "__main__":
    df_all, threshold = load_and_process_data(DATA_PATH)
    model = train_model(df_all, MODEL_PATH)

    df_last = get_last_day_data(df_all)
    status, prob, daily_kwh = predict_efficiency(df_last, MODEL_PATH, threshold_kwh=threshold)
    advice = generate_advice(status, prob, daily_kwh, threshold, df_last)
