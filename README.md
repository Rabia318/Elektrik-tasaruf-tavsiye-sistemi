# Enerji Verimliliği Tahmin ve Öneri Sistemi

Bu proje, farklı ev cihazlarının (Bulaşık Makinesi, Çamaşır Makinesi, Buzdolabı, TV) enerji tüketim verilerini analiz ederek verimlilik tahmini yapar ve kullanıcıya öneriler sunar.

## Özellikler

- PostgreSQL veritabanından 5 dakikalık enerji tüketim verilerini çeker.
- Makine öğrenmesi modelleri ile cihazların enerji verimliliğini tahmin eder.
- Verimlilik durumuna göre öneriler üretir.
- Flask tabanlı API ile tahmin sonuçlarına erişim sağlar.
- `run.py` dosyası ile tüm cihazlar için tahmin ve öneriler konsola yazdırılır.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
