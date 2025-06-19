import joblib
from utils.common_utils import get_device_dataframe
import pandas as pd

MODEL_PATH = "models/camasir_model.pkl"

def load_and_prepare_data():
    print("📥 SQL'den çamaşır makinesi verisi çekiliyor ve işleniyor...")
    df = get_device_dataframe("camasir")

    df["datetime"] = df["timestamp"]
    df["date"] = df["datetime"].dt.date

    daily_consumption = df.groupby("date")["power_watt"].sum()
    threshold = daily_consumption.mean()

    df["Inefficient"] = df["date"].map(lambda d: int(daily_consumption[d] > threshold))

    return df, threshold

def get_last_day_data(df):
    last_date = df["date"].max()
    df_last = df[df["date"] == last_date]
    print(f"📆 Son gün verisi: {last_date}, kayıt sayısı: {len(df_last)}")
    return df_last

def predict_efficiency(df_day, threshold, model_path=MODEL_PATH, threshold_prob=0.1, threshold_daily_factor=1.5):
    print("🤖 Model ile verimlilik tahmini yapılıyor...")

    feature_cols = [col for col in df_day.columns if col not in ["timestamp", "datetime", "date", "Inefficient"]]
    X_avg = df_day[feature_cols].mean().to_frame().T

    model = joblib.load(model_path)
    proba = model.predict_proba(X_avg)[0][1]
    print(f"🔎 Inefficient olasılığı: {proba:.3f}")

    daily_kwh = df_day["power_watt"].sum()

    if proba > threshold_prob or (daily_kwh > threshold * threshold_daily_factor):
        status = "verimsiz"
    else:
        status = "verimli"

    return status, proba, daily_kwh

def generate_advice(status, prob, daily_kwh, threshold, df_day):
    print(f"\n📊 Günlük tüketim: {daily_kwh:.2f} Wh")
    print(f"🎯 Verimlilik durumu: {status.capitalize()}")

    if status == "verimsiz":
        if prob >= 0.8:
            msg = "🚨 Çamaşır makinesi çok sık çalıştırılıyor. Tam kapasiteyle yıkama yapmanız tavsiye edilir."
        elif "is_peak_hour" in df_day.columns and df_day["is_peak_hour"].sum() > len(df_day) * 0.3:
            msg = "⏱️ Yoğun saatlerde çalıştırılmış. Enerji tasarrufu için gece veya sabah saatleri tercih edilebilir."
        else:
            msg = "⚠️ Ortalama tüketim yüksek. Daha kısa veya düşük sıcaklıklı programlar tercih edilebilir."
    else:
        if df_day["power_watt"].max() > threshold * 0.1:
            msg = "ℹ️ Bazı anlarda ani tüketim artışı var. Sık kısa programlar kullanılıyor olabilir."
        else:
            msg = "✅ Çamaşır makinesi verimli çalışıyor. Mevcut kullanım uygundur."

    print(f"💡 Öneri: {msg}")
    return msg

def run_camasir_advice():
    df_all, threshold = load_and_prepare_data()
    df_last = get_last_day_data(df_all)
    status, prob, daily_kwh = predict_efficiency(df_last, threshold)
    advice = generate_advice(status, prob, daily_kwh, threshold, df_last)
    return {
        "status": status,
        "probability": prob,
        "daily_consumption": daily_kwh,
        "advice": advice,
    }
