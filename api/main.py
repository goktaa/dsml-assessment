"""Tahmin servisi.

Çalıştırmak için: uvicorn main:app --app-dir api
"""

import json
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from schemas import OgrenciGirdi, TahminCevabi
from sklearn.exceptions import InconsistentVersionWarning

MODEL_KLASORU = Path(__file__).resolve().parent.parent / "models"

durum: dict = {}


@asynccontextmanager
async def yasam_dongusu(app: FastAPI):
    # Model açılışta bir kez yükleniyor, her istekte değil.
    model_yolu = MODEL_KLASORU / "model.joblib"
    if not model_yolu.exists():
        raise RuntimeError(f"Model yok: {model_yolu}")

    meta = json.loads((MODEL_KLASORU / "meta.json").read_text(encoding="utf-8"))

    # Sadece sürüm uyuşmazlığını ayıklıyorum. Yüklerken gelen diğer uyarılar
    # (numpy'ın kullanımdan kaldırma notları gibi) tahmini etkilemiyor.
    with warnings.catch_warnings(record=True) as yakalanan:
        warnings.simplefilter("always")
        model = joblib.load(model_yolu)
    surum_uyarilari = [str(u.message) for u in yakalanan
                       if issubclass(u.category, InconsistentVersionWarning)]

    durum.update(model=model, meta=meta, surum_uyarilari=surum_uyarilari)
    print(f"Model yüklendi: {meta['model_adi']} v{meta['surum']}")
    print(f"Eğitim ortamı: {meta['ortam']}")
    for u in surum_uyarilari:
        print(f"SÜRÜM UYUŞMAZLIĞI: {u}")
    yield
    durum.clear()


app = FastAPI(
    title="Öğrenci Performans Skoru Tahmini",
    description="Öğrencinin ham kaydını alır, performans skorunu tahmin eder.",
    version="1.0.0",
    lifespan=yasam_dongusu,
)


@app.get("/health", summary="Sağlık kontrolü")
def saglik():
    """Servis ve model durumu."""
    if "model" not in durum:
        raise HTTPException(status_code=503, detail="Model yüklü değil")
    meta = durum["meta"]
    return {
        "durum": "calisiyor",
        "model_adi": meta["model_adi"],
        "model_surumu": meta["surum"],
        "egitim_tarihi": meta["egitim_tarihi"],
        "oznitelik_sirasi": meta["oznitelik_sirasi"],
        "dogrulanmis_metrikler": meta["dogrulanmis_metrikler"],
        "surum_uyarilari": durum["surum_uyarilari"],
    }


@app.post("/predict", response_model=TahminCevabi, summary="Tahmin")
def tahmin_et(girdi: OgrenciGirdi):
    """Bir öğrenci için skor tahmini."""
    if "model" not in durum:
        raise HTTPException(status_code=503, detail="Model yüklü değil")

    model, meta = durum["model"], durum["meta"]
    try:
        X = girdi.oznitelik_satiri(meta)
    except ValueError as hata:
        raise HTTPException(status_code=422, detail=str(hata)) from hata

    tahmin = float(model.predict(X)[0])

    # Aralığı önce ağaçların dağılımından üretiyordum. Nihai model ElasticNet
    # olunca o yöntem çalışmaz hale geldi: doğrusal modelin içinde birbirinden
    # ayrışan ağaçlar yok, tek bir denklem var. Eski kod burada estimators_
    # alanını arayıp AttributeError veriyordu.
    #
    # Yerine doğrulanmış test hatasından sabit bir bant üretiyorum. 1.2816
    # katsayısı normal dağılımda %80'lik aralığa denk geliyor, yani eski
    # 10-90 persentil bandıyla aynı kapsama.
    #
    # Bilinen sınırı: bu bant her öğrenci için aynı genişlikte. Ağaç sürümü
    # "bu öğrenci konusunda kararsızım" diyebiliyordu, bu diyemiyor. Öğrenciye
    # özel belirsizliği aşağıdaki ekstrapolasyon uyarısı taşıyor.
    rmse = float(meta["dogrulanmis_metrikler"]["rmse"])
    yari_genislik = 1.2816 * rmse
    alt, ust = tahmin - yari_genislik, tahmin + yari_genislik

    uyarilar = []
    bos_alan = sum(getattr(girdi, a) is None for a in
                   ["burslu", "aile_geliri_seviyesi", "kardes",
                    "anne_egitim_seviyesi", "baba_egitim_seviyesi",
                    "anne_meslek", "baba_meslek"])
    if bos_alan >= 4:
        uyarilar.append(f"Aile anketinin {bos_alan}/7 alanı boş, dayanak zayıf.")

    # Ekstrapolasyon uyarısı. Modeli zaten ekstrapolasyon davranışı yüzünden
    # seçtim: ağaç modelleri eğitim aralığının dışında sabit cevap veriyordu,
    # ElasticNet düzgün devam ediyor. Yine de tahminin gözlenmemiş bir bölgeden
    # geldiğini söylemek gerekiyor, sessizce üretip geçmek doğru olmaz.
    araliklar = meta.get("oznitelik_araliklari", {})
    disarida = []
    for ad, deger in zip(meta["oznitelik_sirasi"], X.iloc[0]):
        sinir = araliklar.get(ad)
        if sinir is None or deger is None:
            continue
        if deger < sinir[0] or deger > sinir[1]:
            disarida.append(f"{ad}={deger:g} (eğitim aralığı "
                            f"{sinir[0]:g}-{sinir[1]:g})")
    if disarida:
        uyarilar.append(
            "Bu öğrenci eğitimde görülmemiş bir değer aralığında: "
            + "; ".join(disarida)
            + ". Tahmin ekstrapolasyondur, dikkatli yorumlanmalı.")
    if durum["surum_uyarilari"]:
        uyarilar.append("Model farklı bir kütüphane sürümüyle üretilmiş.")

    return TahminCevabi(
        tahmin=round(tahmin, 2),
        guven_araligi=[round(float(alt), 2), round(float(ust), 2)],
        model_adi=meta["model_adi"],
        model_surumu=meta["surum"],
        bos_birakilan_anket_alani=bos_alan,
        uyarilar=uyarilar,
    )
