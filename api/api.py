from flask import Flask, request, jsonify
import pandas as pd
from file_utils import load_data, get_last_day_data
from predictor import load_model, predict_efficiency, generate_advice

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        device_type = data.get("device_type")
        df_json = data.get("data")  # Mobil uygulamadan gelen cihaz verisi JSON listesi

        if not device_type or not df_json:
            return jsonify({"error": "device_type veya data alanı eksik"}), 400

        df = pd.DataFrame(df_json)
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["datetime"].dt.date

        # Son gün verisi alınabilir veya tüm data ile tahmin yapılabilir
        df_day = get_last_day_data(df)

        model = load_model(device_type)

        # Burada opsiyonel olarak eşik değeri de alabiliriz, şimdi sabit
        status, prob, daily_kwh = predict_efficiency(df_day, model)

        # Öneri mesajı oluştur
        threshold = df_day["power_watt"].mean()  # basit eşik olarak günlük ortalama kullandım
        advice = generate_advice(device_type, status, prob, daily_kwh, threshold, df_day)

        result = {
            "status": status,
            "probability": prob,
            "daily_kwh": daily_kwh,
            "advice": advice
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
