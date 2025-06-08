import joblib

# Model dosyalarının yolunu cihaz tipine göre tanımla
MODEL_PATHS = {
    "buzdolabi": r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/mühp2/models/buzdolabi_model.pkl",
    "bulasik": r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/mühp2/models/bulasik_model.pkl",
    "camasir": r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/mühp2/models/camasir_model.pkl",
    "televizyon": r"C:/Users/casper/Desktop/Class_3_sprng/Müh Pro/Veri/veri_setleri/mühp2/models/televizyon_model.pkl",
}

def load_model(device_type):
    """
    Cihaz tipine göre ilgili modeli yükler.
    """
    model_path = MODEL_PATHS.get(device_type)
    if not model_path:
        raise ValueError(f"Model path for device '{device_type}' not found.")
    model = joblib.load(model_path)
    return model

def predict_efficiency(df_day, model, threshold_prob=0.1, threshold_daily_factor=1.5, threshold_kwh=None):
    """
    Günlük veri üzerinden verimlilik tahmini yapar.
    
    Args:
      df_day: Günlük veri DataFrame'i (son gün verisi)
      model: Yüklü sklearn modeli
      threshold_prob: Verimsiz olma olasılığı eşiği
      threshold_daily_factor: Günlük tüketim karşılaştırma faktörü
      threshold_kwh: Günlük tüketim için referans eşik (opsiyonel)
    
    Returns:
      status: 'verimli' veya 'verimsiz'
      proba: Verimsiz olma olasılığı (0-1)
      daily_kwh: Günlük toplam tüketim (Wh)
    """

    feature_cols = [col for col in df_day.columns if col not in ["timestamp", "datetime", "date", "Inefficient"]]
    # Günlük ortalama feature seti alınıyor
    X_avg = df_day[feature_cols].mean().to_frame().T

    # Modelden verimsiz olma olasılığı hesaplanıyor
    proba = model.predict_proba(X_avg)[0][1]

    # Günlük toplam tüketim (Wh)
    daily_kwh = df_day["power_watt"].sum()

    # Eşiklere göre verimlilik durumu belirleniyor
    if proba > threshold_prob or (threshold_kwh is not None and daily_kwh > threshold_kwh * threshold_daily_factor):
        status = "verimsiz"
    else:
        status = "verimli"

    return status, proba, daily_kwh

def generate_advice(device_type, status, prob, daily_kwh, threshold, df_day):
    """
    Verimlilik durumuna göre cihaz bazlı öneri metni döner.
    
    Args:
      device_type: cihaz türü (ör: buzdolabi, bulasik, televizyon, ...)
      status: 'verimli' veya 'verimsiz'
      prob: modelden dönen verimsizlik olasılığı
      daily_kwh: günlük toplam tüketim (Wh)
      threshold: günlük tüketim için referans eşik
      df_day: günlük veri DataFrame'i (opsiyonel kriterler için)
    
    Returns:
      String olarak öneri mesajı
    """
    advice_dict = {
        "buzdolabi": {
            "verimsiz_high_prob": "🚨 Buzdolabı kapakları çok sık açılıp kapanıyor. Sıcaklığın sabit kalması için dikkat edin.",
            "verimsiz_peak_use": "⏱️ Puant saatlerde buzdolabı kullanımı yüksek. Elektrik faturası için düşük tarifeli saatler tercih edin.",
            "verimsiz_default": "⚠️ Ortalama tüketim yüksek. Buzdolabının içini fazla doldurmamaya ve düzenli temizlemeye dikkat edin.",
            "verimli_high_peak": "ℹ️ Gün içinde kısa süreli yüksek tüketim gözlendi. Buzdolabı iyi çalışıyor.",
            "verimli_default": "✅ Buzdolabı verimli kullanılıyor. Mevcut kullanım alışkanlıklarınız uygun."
        },
        "bulasik": {
            "verimsiz_high_prob": "🚨 Bulaşık makinesi çok sık çalıştırılıyor. Tam dolu çalıştırmayı tercih edin.",
            "verimsiz_peak_use": "⏱️ Puant saatlerde bulaşık makinesi kullanımı yüksek. Daha uygun saatlerde çalıştırmayı deneyin.",
            "verimsiz_default": "⚠️ Ortalama tüketim yüksek. Hızlı programlar yerine standart programlar tercih edilebilir.",
            "verimli_high_peak": "ℹ️ Bazı saatlerde tüketim yüksek, kısa süreli kullanımlar olabilir.",
            "verimli_default": "✅ Bulaşık makinesi verimli çalışıyor. Kullanım alışkanlıklarınız iyi."
        },
        "camasir": {
            "verimsiz_high_prob": "🚨 Çamaşır makinesi çok sık çalıştırılıyor. Tam dolu çalıştırmayı tercih edin.",
            "verimsiz_peak_use": "⏱️ Puant saatlerde çamaşır makinesi kullanımı yüksek. Düşük tarifeli saatlerde çalıştırmayı deneyin.",
            "verimsiz_default": "⚠️ Ortalama tüketim yüksek. Düşük sıcaklık programları tercih edin.",
            "verimli_high_peak": "ℹ️ Kısa süreli yüksek tüketim olabilir.",
            "verimli_default": "✅ Çamaşır makinesi verimli kullanılıyor."
        },
        "televizyon": {
            "verimsiz_high_prob": "🚨 TV çok uzun süre açık kalıyor. Kullanım süresini azaltmayı düşünün.",
            "verimsiz_peak_use": "⏱️ TV puant saatlerde sık kullanılıyor. Daha düşük tarifeli saatlerde izlemeye çalışın.",
            "verimsiz_default": "⚠️ Ortalama tüketim yüksek. TV'nin gereksiz açık bırakılmadığından emin olun.",
            "verimli_high_peak": "⚠️ Bazı saatlerde TV'nin tüketimi çok yüksek, arka planda açık kalmış olabilir.",
            "verimli_default": "✅ Televizyon verimli çalışıyor. Mevcut kullanım şekli uygundur."
        }
    }

    adv = advice_dict.get(device_type, {})

    if status == "verimsiz":
        if prob >= 0.8:
            msg = adv.get("verimsiz_high_prob", "Verimsiz kullanım tespit edildi.")
        elif "is_peak" in df_day.columns and df_day["is_peak"].sum() > len(df_day) * 0.3:
            msg = adv.get("verimsiz_peak_use", "Puant saatlerde yüksek tüketim var.")
        else:
            msg = adv.get("verimsiz_default", "Verimsiz kullanım gözlendi.")
    else:
        if df_day["power_watt"].max() > threshold * 0.1:
            msg = adv.get("verimli_high_peak", "Kısa süreli yüksek tüketim olabilir.")
        else:
            msg = adv.get("verimli_default", "Verimli kullanım mevcut.")

    return msg
