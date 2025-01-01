import sqlite3
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import pickle
import os
from def_file import *


# Modelleri indir ve kaydet
#model_download()
# GPU veya CPU cihazını kontrol et
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modeli yükle ve GPU'ya taşı
model = SentenceTransformer('bert-base-nli-mean-tokens')
model = model.to(device)

# Veritabanına bağlan
conn = sqlite3.connect("ElectionDb.db")
query = "SELECT id, Corpus FROM elcPart ORDER BY id DESC"
df = pd.read_sql_query(query, conn)

# İşlem parametreleri
batch_size = 10  # Kaç satırın aynı anda işleneceğini belirler
total_rows = len(df)
num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)

# Verileri batch halinde işleme
for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, total_rows)
    batch = df.iloc[start_idx:end_idx]

    data_to_update = []  # Bu batch için veri listesi
    for index, row in batch.iterrows():
        sentence = row["Corpus"]
        embedding = model.encode(sentence)  # Vektör hesaplama

        # Vektörleri BLOB formatına dönüştür
        embedding_blob = pickle.dumps(embedding)
        data_to_update.append((embedding_blob, row["id"]))

    # Veritabanını güncelle
    conn.executemany("UPDATE elcPart SET Vector3 = ? WHERE id = ?", data_to_update)
    conn.commit()  # Her batch'ten sonra commit

    print(f"Batch {batch_num + 1}/{num_batches} tamamlandı: {len(data_to_update)} kayıt işlendi.")

conn.close()
print("Tüm veriler başarıyla güncellendi!")


