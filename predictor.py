import joblib
from utils.common_utils import get_device_dataframe

MODEL_PATHS = {
    "bulasik": "models/bulasik_model.pkl",
    "camasir": "models/camasir_model.pkl",
    "buzdolabi": "models/buzdolabi_model.pkl",
    "tv": "models/tv_model.pkl",
}

def load_model(device_type):
    model_path = MODEL_PATHS.get(device_type)
    if not model_path:
        raise ValueError(f"{device_type} için model bulunamadı.")
    return joblib.load(model_path)

def prepare_data(device_type):
    df = get_device_dataframe(device_type)
    df["datetime"] = df["timestamp"]
    df["date"] = df["datetime"].dt.date

    daily_consumption = df.groupby("date")["power_watt"].sum()
    threshold = daily_consumption.mean()
    df["Inefficient"] = df["date"].map(lambda d: int(daily_consumption[d] > threshold))

    return df, threshold

def get_last_day_data(df):
    last_date = df["date"].max()
    return df[df["date"] == last_date]

def predict_efficiency(df_day, model, threshold, threshold_prob=0.1, threshold_daily_factor=1.5):
    feature_cols = [c for c in df_day.columns if c not in ["timestamp", "datetime", "date", "Inefficient"]]
    X_avg = df_day[feature_cols].mean().to_frame().T

    proba = model.predict_proba(X_avg)[0][1]
    daily_kwh = df_day["power_watt"].sum()

    if proba > threshold_prob or (daily_kwh > threshold * threshold_daily_factor):
        status = "verimsiz"
    else:
        status = "verimli"

    return status, proba, daily_kwh

def generate_advice(device_type, status, prob, daily_kwh, threshold, df_day):
    # Burada cihaz tipine göre tavsiyeler veriyoruz
    if device_type == "bulasik":
        if status == "verimsiz":
            if prob >= 0.8:
                msg = "🚨 Bulaşık makinesi çok sık kullanılıyor. Tam dolmadan çalıştırmayın."
            elif df_day["is_peak_hour"].sum() > len(df_day) * 0.3:
                msg = "⏱️ Puant saatlerde yıkama yapılmış. Gece saatleri tercih edilebilir."
            else:
                msg = "⚠️ Ortalama tüketim yüksek. Eko mod tercih edin."
        else:
            msg = "✅ Bulaşık makinesi verimli kullanılıyor."

    elif device_type == "camasir":
        if status == "verimsiz":
            if prob >= 0.8:
                msg = "🚨 Çamaşır makinesi çok sık çalıştırılıyor. Tam kapasite kullanın."
            elif df_day["is_peak_hour"].sum() > len(df_day) * 0.3:
                msg = "⏱️ Yoğun saatlerde kullanılmış. Gece veya sabah tercih edin."
            else:
                msg = "⚠️ Ortalama tüketim yüksek. Daha kısa/düşük sıcaklık programı seçin."
        else:
            msg = "✅ Çamaşır makinesi verimli çalışıyor."

    elif device_type == "buzdolabi":
        if status == "verimsiz":
            if prob >= 0.8:
                msg = "🚨 Buzdolabı çok yüksek tüketiyor. Motoru kontrol edin."
            elif df_day["is_peak_hour"].sum() > len(df_day) * 0.3:
                msg = "⏱️ Puant saatlerde yüksek tüketim var. Tarife kontrolü yapılmalı."
            else:
                msg = "⚠️ Ortalama tüketim yüksek. İç sıcaklık ayarlarını kontrol edin."
        else:
            msg = "✅ Buzdolabı verimli çalışıyor."

    elif device_type == "tv":
        if status == "verimsiz":
            if prob >= 0.8:
                msg = "🚨 TV çok uzun süre açık kalıyor. Kullanım süresini azaltın."
            elif df_day["is_peak_hour"].sum() > len(df_day) * 0.3:
                msg = "⏱️ TV puant saatlerde sık kullanılıyor. Düşük tarifeli saatleri tercih edin."
            else:
                msg = "⚠️ Ortalama tüketim yüksek. Gereksiz açık bırakmayın."
        else:
            msg = "✅ Televizyon verimli kullanılıyor."

    else:
        msg = "Bilinen cihaz tipi değil, tavsiye verilemiyor."

    return msg
