from utils import bulasik_utils, camasir_utils, buzdolabi_utils, tv_utils

def main():
    print("🔋 Enerji Verimliliği Tahmin ve Öneri Sistemi\n")

    cihazlar = {
        "Bulaşık Makinesi": bulasik_utils.run_bulasik_advice,
        "Çamaşır Makinesi": camasir_utils.run_camasir_advice,
        "Buzdolabı": buzdolabi_utils.run_buzdolabi_advice,
        "TV": tv_utils.run_tv_advice,
    }

    for cihaz_adi, run_func in cihazlar.items():
        print(f"--- {cihaz_adi} ---")
        try:
            result = run_func()
            print(f"Durum: {result['status']}")
            print(f"İlgili Olasılık: {result['probability']:.3f}")
            print(f"Günlük Tüketim (Wh): {result['daily_consumption']:.2f}")
            print(f"Öneri: {result['advice']}\n")
        except Exception as e:
            print(f"⚠️ {cihaz_adi} için tahmin çalıştırılamadı: {e}\n")

if __name__ == "__main__":
    main()
