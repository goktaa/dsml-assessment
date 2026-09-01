# Öğrenci Performans Skoru Tahmini

Öğrenci kayıtlarından `hedef_degisken` tahmin eden bir model ve onu servis eden
bir API.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **ElasticNet (seçilen)** | **31.993** | **24.957** | **0.7510** |
| RandomForest | 29.139 | 23.194 | 0.7934 |
| Aptal referans (ortalamayı söyler) | 64.112 | 50.556 | -0.0001 |

Kronolojik doğrulama: dönem 0+1 ile eğitim, dönem 2 ile test (108 öğrenci).

En düşük RMSE'li modeli seçmedim. RandomForest 2.85 puan daha iyi ama iki
testte kayboluyor:

- **Ekstrapolasyon:** eğitimde görülen `genel_not_ort` aralığının (22.2-99.7)
  dışında RandomForest'ın tahmini hiç değişmiyor (0.00 puan), çünkü ağaç en uç
  yaprağını tekrarlıyor. ElasticNet dışarıda da düzgün devam ediyor (72.26 puan).
- **Ezberleme makası:** RandomForest eğitimde 24.35, testte 29.50 hata yapıyor
  (+5.15). ElasticNet'te makas negatif (-2.29), yani ezberleme yok.

Ayrıca ElasticNet 30.5 KB / 0.87 ms, RandomForest 4.3 MB / 61 ms. Edge cihazda
çalışacak bir servis için bu fark belirleyici.

## Servisi çalıştırma

Model `models/` klasöründe hazır geliyor, notebook'u çalıştırmadan servis
ayağa kalkıyor.

### 1. Başlat

```bash
docker compose up -d --build
```

### 2. Sağlık kontrolü

```bash
curl http://localhost:8000/health
```

### 3. Tahmin

Örnek istek:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "turkce_not_ort": 78.5, "matematik_not_ort": 82.0, "fen_not_ort": 85.3,
    "sosyal_not_ort": 87.0, "y_dil_not_ort": 84.0, "din_not_ort": 87.5,
    "burslu": 0, "aile_geliri_seviyesi": 2, "kardes": 0,
    "anne_egitim_seviyesi": 1, "baba_egitim_seviyesi": 1,
    "anne_meslek": 0, "baba_meslek": 1,
    "dogum_yil": 2005, "okula_basladigi_yil": "2015-2016"
  }'
```

Cevabı:

```json
{
  "tahmin": 355.95,
  "guven_araligi": [320.23, 394.13],
  "model_adi": "ElasticNet",
  "model_surumu": "1.0.0",
  "bos_birakilan_anket_alani": 0,
  "uyarilar": []
}
```

Tarayıcıdan denemek için `/docs`, durdurmak için `docker compose down`.

### Girdi

Servis ham alanları alıyor. Not ortalaması, ebeveyn eğitim ortalaması gibi
türetilmiş değerleri kendisi hesaplıyor, dışarıdan beklemiyor. Böylece formül
tek yerde duruyor ve eğitimle servis arasında fark oluşmuyor.

Aile anketi alanları (`burslu`, `aile_geliri_seviyesi`, `kardes`,
`anne_egitim_seviyesi`, `baba_egitim_seviyesi`, `anne_meslek`, `baba_meslek`)
boş gelebilir. Boşluk hata değil, modelin kullandığı bir bilgi: anketi
doldurmayan ailelerin öğrencileri ortalamanın anlamlı biçimde altında. Ders
notlarından en az biri gerekiyor, hiç not yoksa geriye modelin ortalamayı
söylemesi kalıyor.

## Notebook

Analizin tamamı tek dosyada: `notebooks/01_explore_data.ipynb`. EDA, temizlik,
öznitelik mühendisliği, modelleme ve modelin dışa aktarılması aynı akışta.

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_explore_data.ipynb
```

Baştan sona çalışması 25-30 dakika sürüyor (çapraz doğrulamalar ve
GridSearchCV). Son bölüm `models/model.joblib` ve `models/meta.json`
dosyalarını yeniden üretiyor.

## Proje yapısı

```
notebooks/01_explore_data.ipynb   analizin tamamı
models/                           eğitilmiş model + meta.json
api/                              FastAPI servisi ve Dockerfile
docker-compose.yml                servisi ayağa kaldırma
data/raw/data.db                  kaynak veri (SQLite)
```

## Notlar

- Sinyalin büyük kısmı tek bir öznitelikten geliyor (`genel_not_ort`, SHAP
  katkısının büyük çoğunluğu). Model esasen "notu yüksek olan öğrenci yüksek
  skor alır" diyor.
- `guven_araligi`, doğrulanmış test hatasından türetilmiş sabit bant:
  tahmin ± 1.2816 × RMSE, yaklaşık %80'lik aralık. İstatistiksel bir güven
  aralığı değil. Her öğrenci için aynı genişlikte; öğrenciye özel belirsizliği
  ekstrapolasyon uyarısı taşıyor.
- `cinsiyet` ve `il` modele girmiyor. ElasticNet'in L1 cezası
  `aile_geliri_seviyesi` dahil üç özniteliği tamamen sıfırlıyor, yani model
  aile gelirini doğrudan kullanmıyor.
- Öznitelik seçimi üç ölçütle doğrulandı: korelasyon, RandomForest önemi ve
  SHAP. Açıklanabilirlik için de SHAP kullanılıyor.
- `requirements-api.txt` sürümleri sabit. `model.joblib` bir pickle, farklı
  scikit-learn sürümünde düzgün yüklenmiyor.
- Test kümesi 108 öğrenci. Modeller arası 1 puanlık farkları ayırt etmeye
  yetmiyor.
