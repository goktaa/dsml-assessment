# Öğrenci Performans Skoru Tahmini

Öğrenci kayıtlarından `hedef_degisken` tahmin eden bir model ve onu servis eden
bir API.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| RandomForest | 29.139 | 23.194 | 0.7934 |
| Aptal referans (ortalamayı söyler) | 64.112 | 50.556 | -0.0001 |

Kronolojik doğrulama: dönem 0+1 ile eğitim, dönem 2 ile test (108 öğrenci).

## Nasıl Kullanılır

### 1. Servisi başlat

```bash
docker compose up -d --build
```

İlk kurulumda imaj indirileceği için birkaç dakika sürer, sonrasında saniyeler
içinde açılır. Model `models/` klasöründe hazır geliyor, notebook'u çalıştırmanız
gerekmiyor.

### 2. Ayakta mı, kontrol et

```bash
curl http://localhost:8000/health
```

### 3. Tahmin al

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

```json
{
  "tahmin": 355.95,
  "guven_araligi": [320.23, 394.13],
  "model_adi": "RandomForest",
  "model_surumu": "1.0.0",
  "bos_birakilan_anket_alani": 0,
  "uyarilar": []
}
```

Tarayıcıdan denemek için: <http://localhost:8000/docs>

Durdurmak için: `docker compose down`

### Girdi hakkında bilinmesi gereken iki şey

**Ham alanlar gönderiliyor.** Not ortalaması, ebeveyn eğitim ortalaması gibi
türetilmiş değerleri servis kendisi hesaplıyor, sizin göndermenize gerek yok.

**Aile anketi alanları boş bırakılabilir.** `burslu`, `aile_geliri_seviyesi`,
`kardes`, `anne_egitim_seviyesi`, `baba_egitim_seviyesi`, `anne_meslek`,
`baba_meslek` alanlarını atlayabilirsiniz. Boşluk hata değil, modelin kullandığı
bir bilgi. Ders notlarından ise en az biri zorunlu.

## Notebook

Analizin tamamı tek dosyada: `notebooks/01_explore_data.ipynb`.

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_explore_data.ipynb
```

Baştan sona çalıştırmak 25-30 dakika sürüyor (çapraz doğrulamalar ve
GridSearchCV). Son bölüm `models/model.joblib` ve `models/meta.json` dosyalarını
yeniden üretiyor.

## Proje yapısı

```
notebooks/01_explore_data.ipynb   EDA, temizlik, öznitelik mühendisliği,
                                  modelleme, model dışa aktarma
models/                           Eğitilmiş model + meta.json
api/                              FastAPI servisi ve Dockerfile
docker-compose.yml                Servisi ayağa kaldırma
data/raw/data.db                  Kaynak veri (SQLite)
```

## Notlar

- Sinyalin büyük kısmı tek bir öznitelikten geliyor (`genel_not_ort`,
  RandomForest öneminin %82'si). Model esasen "notu yüksek olan öğrenci yüksek
  skor alır" diyor.
- `guven_araligi`, ormandaki 300 ağacın %10-%90 aralığı. Modelin kendi
  içindeki anlaşmazlığın ölçüsü, istatistiksel bir güven aralığı değil.
- `cinsiyet` ve `il` modele girmiyor.
- `requirements-api.txt` sürümleri sabit: `model.joblib` bir pickle, farklı
  scikit-learn sürümünde düzgün yüklenmiyor.
