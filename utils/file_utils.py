import pandas as pd

def load_data(file_path):
    """Excel dosyasını oku ve datetime sütunlarını işle."""
    df = pd.read_excel(file_path)

    # Zaman bilgisi dönüştürme
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["datetime"].dt.date

    return df

def get_last_day_data(df):
    """Verinin en son gününe ait verileri döner."""
    last_date = df["date"].max()
    df_last_day = df[df["date"] == last_date].copy()
    return df_last_day
