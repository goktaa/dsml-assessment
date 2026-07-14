import sqlite3
import pandas as pd

conn = sqlite3.connect("../data/raw/data.db")

# Her tablonun ilk 5 satirini goster
for table_name in ["ogrenci", "not", "hedef"]:
    print(f"\n--- {table_name} ---")
    df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 5', conn)
    print(df)
    count = pd.read_sql(f'SELECT COUNT(*) as cnt FROM "{table_name}"', conn)['cnt'][0]
    print(f"Toplam satir sayisi: {count}")
    print("\n--- not tablosu veri tipleri ---")
df_not = pd.read_sql('SELECT * FROM "not"', conn)
print(df_not.dtypes)
schema = pd.read_sql('PRAGMA table_info("not")', conn)
print(schema)
# %%
print(df_not['turkce_not_ort'].unique()[:20])
# %%
print(df_not.isnull().sum())
# %%
df_not['turkce_not_ort_temiz'] = df_not['turkce_not_ort'].str.replace(',', '.')
print(df_not['turkce_not_ort_temiz'].unique()[:20])
# %%
df_not['turkce_not_ort_temiz'] = pd.to_numeric(df_not['turkce_not_ort_temiz'], errors='coerce')
print(df_not['turkce_not_ort_temiz'].dtype)
print(df_not['turkce_not_ort_temiz'].isnull().sum())
