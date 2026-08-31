"""Girdi ve çıktı şemaları.

Servis ham alanları alıyor. Not ortalaması, ebeveyn eğitim ortalaması gibi
türetilmiş kolonları burada hesaplıyorum, dışarıdan beklemiyorum. Böylece
formül tek yerde duruyor ve eğitimle servis arasında fark oluşmuyor.
"""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator

# Bu alanların boş olması hata değil. Anketi doldurmayan aile modelin
# kullandığı bir sinyal, o yüzden boşlukları sayıyorum.
AILE_ANKETI = [
    "burslu",
    "aile_geliri_seviyesi",
    "kardes",
    "anne_egitim_seviyesi",
    "baba_egitim_seviyesi",
    "anne_meslek",
    "baba_meslek",
]


class OgrenciGirdi(BaseModel):
    """Bir öğrencinin ham kaydı."""

    # Ders notları
    turkce_not_ort: Optional[float] = Field(None, ge=0, le=100)
    matematik_not_ort: Optional[float] = Field(None, ge=0, le=100)
    fen_not_ort: Optional[float] = Field(None, ge=0, le=100)
    sosyal_not_ort: Optional[float] = Field(None, ge=0, le=100)
    y_dil_not_ort: Optional[float] = Field(None, ge=0, le=100)
    din_not_ort: Optional[float] = Field(None, ge=0, le=100)

    # Aile anketi, boş bırakılabilir
    burslu: Optional[int] = Field(None, ge=0, le=1)
    aile_geliri_seviyesi: Optional[int] = Field(None, ge=0, le=4)
    kardes: Optional[int] = Field(None, ge=0, le=1)
    anne_egitim_seviyesi: Optional[int] = Field(None, ge=0, le=4)
    baba_egitim_seviyesi: Optional[int] = Field(None, ge=0, le=4)
    anne_meslek: Optional[int] = Field(None, ge=0, le=1)
    baba_meslek: Optional[int] = Field(None, ge=0, le=1)

    # Okul kaydı
    dogum_yil: int = Field(..., ge=1980, le=2015)
    okula_basladigi_yil: str = Field(..., examples=["2014-2015"])

    @model_validator(mode="after")
    def en_az_bir_not(self):
        notlar = [self.turkce_not_ort, self.matematik_not_ort, self.fen_not_ort,
                  self.sosyal_not_ort, self.y_dil_not_ort, self.din_not_ort]
        if all(n is None for n in notlar):
            # Sinyalin %82'si not ortalamasından geliyor. Hiç not yoksa geriye
            # kalan şey modelin ortalamayı söylemesi, ona tahmin denmez.
            raise ValueError("En az bir ders notu gerekli.")
        return self

    def oznitelik_satiri(self, meta: dict) -> pd.DataFrame:
        """Ham alanlardan modelin beklediği 10 kolonu üretir.

        Kolon sırası meta.json'dan geliyor, elle yazmıyorum: sklearn kolon
        ismine değil sıraya bakıyor.
        """
        ders = {
            "turkce_not_ort": self.turkce_not_ort,
            "matematik_not_ort": self.matematik_not_ort,
            "fen_not_ort": self.fen_not_ort,
            "sosyal_not_ort": self.sosyal_not_ort,
            "y_dil_not_ort": self.y_dil_not_ort,
            "din_not_ort": self.din_not_ort,
        }
        dolu = [v for v in ders.values() if v is not None]
        genel_not_ort = sum(dolu) / len(dolu) if dolu else None

        sayisal = [v for v in (self.matematik_not_ort, self.fen_not_ort) if v is not None]
        sozel = [v for v in (self.turkce_not_ort, self.sosyal_not_ort,
                             self.y_dil_not_ort, self.din_not_ort) if v is not None]
        if sayisal and sozel:
            sayisal_sozel_farki = sum(sayisal) / len(sayisal) - sum(sozel) / len(sozel)
        else:
            sayisal_sozel_farki = None

        # Tek ebeveyn biliniyorsa onu kullanıyorum, ortalama alacak ikinci
        # değer yok diye satırı boşa düşürmek gereksiz.
        anne, baba = self.anne_egitim_seviyesi, self.baba_egitim_seviyesi
        if anne is not None and baba is not None:
            ebeveyn_egitim_ort = (anne + baba) / 2
        elif anne is not None:
            ebeveyn_egitim_ort = float(anne)
        elif baba is not None:
            ebeveyn_egitim_ort = float(baba)
        else:
            ebeveyn_egitim_ort = None

        eksik_alan_sayisi = sum(getattr(self, alan) is None for alan in AILE_ANKETI)

        harita = meta.get("okula_basladigi_yil_haritasi") or {}
        yil_kodu = harita.get(self.okula_basladigi_yil)
        if yil_kodu is None:
            raise ValueError(
                f"Bilinmeyen okula_basladigi_yil: {self.okula_basladigi_yil!r}. "
                f"Beklenen: {sorted(harita)}")

        degerler = {
            "genel_not_ort": genel_not_ort,
            "ebeveyn_egitim_ort": ebeveyn_egitim_ort,
            "sayisal_sozel_farki": sayisal_sozel_farki,
            "aile_geliri_seviyesi": self.aile_geliri_seviyesi,
            "anne_meslek(var/yok)": self.anne_meslek,
            "kardes(var/yok)": self.kardes,
            "dogum_yil": self.dogum_yil,
            "burslu": self.burslu,
            "okula_basladigi_yil": yil_kodu,
            "eksik_alan_sayisi": eksik_alan_sayisi,
        }
        sira = meta["oznitelik_sirasi"]
        eksik_kolon = [k for k in sira if k not in degerler]
        if eksik_kolon:
            raise ValueError(f"Üretilemeyen kolon: {eksik_kolon}")
        return pd.DataFrame([[degerler[k] for k in sira]], columns=sira, dtype="float64")


class TahminCevabi(BaseModel):
    tahmin: float
    guven_araligi: list[float]
    model_adi: str
    model_surumu: str
    bos_birakilan_anket_alani: int
    uyarilar: list[str]
