# --- utils/common_utils.py ---
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "molfern",
    "database": "postwebsocket"
}

def get_device_dataframe(device_name):
    # PostgreSQL bağlantısı kur
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # İlgili cihazın ID'sini al
    cursor.execute("SELECT id FROM devices WHERE cihaz_adi = %s", (device_name,))
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"'{device_name}' adlı cihaz bulunamadı.")

    device_id = result[0]

    # Son 1 güne ait verileri çek
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)

    query = """
        SELECT zaman as timestamp, power_watt 
        FROM power_data
        WHERE device_id = %s AND zaman BETWEEN %s AND %s
    """

    df = pd.read_sql(query, conn, params=(device_id, start_time, end_time))
    conn.close()

    if df.empty:
        raise ValueError("Seçilen cihaz için son 1 günde veri bulunamadı.")

    # ------------------ ÖN İŞLEME ------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_only"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour_only"].apply(lambda x: int(x < 6 or x >= 22))
    df["is_peak_hour"] = df["hour_only"].apply(lambda x: int(17 <= x <= 21))
    df["is_business_hour"] = df["hour_only"].apply(lambda x: int(9 <= x <= 17))

    # Enerji tüketimi ve maliyet tahmini
    df["energy_kwh"] = df["power_watt"] / (1000 * 12)  # 5 dakikalık veri
    df["cost_estimate"] = df["energy_kwh"] * 3.0  # Örnek birim fiyat (TL)

    # Time period one-hot encoding
    df["time_period_Day"] = df["hour_only"].apply(lambda x: int(9 <= x < 17))
    df["time_period_Night"] = df["hour_only"].apply(lambda x: int(x < 6 or x >= 22))
    df["time_period_Peak"] = df["hour_only"].apply(lambda x: int(17 <= x <= 21))

    # Günlük tüketim ortalaması üzerinden 'Inefficient' etiketi
    df["date"] = df["timestamp"].dt.date
    daily_consumption = df.groupby("date")["power_watt"].sum()
    threshold = daily_consumption.mean()
    df["Inefficient"] = df["date"].map(lambda d: int(daily_consumption[d] > threshold))

    return df
