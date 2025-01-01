import sqlite3
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
print(str(device))
models = [
    ('./models/bert-base-nli-mean-tokens', 'BERT'),
    ('./models/distilbert-base-nli-stsb-mean-tokens', 'DistilBERT'),
    ('./models/albert-base-v2', 'ALBERT'),
    ('./models/t5-base', 'T5') ,
    ('./models/roberta-base', 'RoBERTa')


]

conn = sqlite3.connect("ElectionDb.db")
cursor = conn.cursor()

required_columns = [
    'Vector_BERT',
    'Vector_DistilBERT',
    'Vector_ALBERT',
    'Vector_RoBERTa',
    'Vector_T5'
]
# Tabloyu kontrol et ve eksik sütunları ekle
def check_and_add_columns():
    cursor.execute("PRAGMA table_info(elcPart);")
    existing_columns = [column[1] for column in cursor.fetchall()]

    for column in required_columns:
        if column not in existing_columns:
            print(f"Sütun ekleniyor: {column}")
            cursor.execute(f"ALTER TABLE elcPart ADD COLUMN {column} BLOB;")
            conn.commit()

check_and_add_columns()
model_instances = {name: SentenceTransformer(model_path).to(device) for model_path, name in models}
query = "SELECT id, Corpus FROM elcPart ORDER BY id DESC"
df = pd.read_sql_query(query, conn)
batch_size = 10
total_rows = len(df)
num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)

for model_name, model in model_instances.items():
    print(f"Model: {model_name} başlatılıyor...")

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch = df.iloc[start_idx:end_idx]

        data_to_update = []

        for index, row in batch.iterrows():
            sentence = row["Corpus"]
            embedding = model.encode(sentence)
            embedding_blob = pickle.dumps(embedding)
            data_to_update.append((embedding_blob, row["id"]))
        cursor.executemany(f"UPDATE elcPart SET Vector_{model_name} = ? WHERE id = ?", data_to_update)
        conn.commit()
        print(f"Model {model_name} Batch {batch_num + 1}/{num_batches} tamamlandı: {len(batch)} kayıt işlendi.")
    print(f"Model {model_name} işlemi tamamlandı.")
conn.close()
print("Tüm veriler başarıyla güncellendi!")
