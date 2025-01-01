import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sqlite3
import pickle

# Model isimleri
VECTOR_COLUMNS = ["Vector_T5", "Vector_BERT", "Vector_DistilBERT", "Vector_RoBERTa", "Vector_ALBERT","Vector_sBERTVector_BERT", "Vector_sBERTVector_DistilBERT", "Vector_sBERTVector_RoBERTa", "Vector_sBERTVector_ALBERT",]

def load_data(db_path, table_name):
    """SQLite veritabanından veriyi yükler."""
    conn = sqlite3.connect(db_path)
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df

def deserialize_vector(pickle_blob):
    """Pickle ile saklanan BLOB vektörünü çözer."""
    return np.array(pickle.loads(pickle_blob))

def apply_pca(vectors, n_components=768):
    """
    PCA ile boyut indirgeme yapar ve veri kaybı oranını hesaplar.
    """
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(vectors)
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    loss_percentage = (1 - explained_variance_ratio) * 100
    return reduced_vectors, loss_percentage

def update_database_column(conn, table_name, column_name, values, ids):
    """
    Veritabanında yeni bir sütun ekler veya mevcut sütunu günceller.
    """
    cursor = conn.cursor()

    # Mevcut sütunları kontrol et
    existing_columns = [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})")]

    # Sütun yoksa ekle
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} BLOB")
        conn.commit()

    # Sütunu güncelle
    for idx, value in zip(ids, values):
        cursor.execute(f"UPDATE {table_name} SET {column_name} = ? WHERE id = ?", (sqlite3.Binary(value), idx))
    conn.commit()

def combine_vectors(df, column1, column2):
    """
    İki sütundaki vektörleri birleştirir.
    """
    combined_vectors = []
    for idx, row in df.iterrows():
        try:
            vec1 = deserialize_vector(row[column1])
            vec2 = deserialize_vector(row[column2])
            combined = np.concatenate([vec1, vec2])
            combined_vectors.append(combined)
        except Exception as e:
            print(f"Vektör birleştirme hatası (İndeks: {idx}): {e}")
    return np.array(combined_vectors)

def process_and_save_combined_vectors(db_path, table_name, vector_columns):
    """
    Veritabanında birleşik vektör sütunları oluşturur ve PCA ile boyut indirgeme uygular.
    """
    conn = sqlite3.connect(db_path)
    df = load_data(db_path, table_name)

    combined_info = []  # PCA ve Loss bilgilerini tutar

    try:
        for col in vector_columns[1:]:
            combined_col_name = f"Vector_T5{col}"
            print(f"{combined_col_name} oluşturuluyor...")

            combined_vectors = combine_vectors(df, vector_columns[0], col)
            if combined_vectors.size == 0:
                print(f"{col} için geçerli birleşim bulunamadı, atlanıyor.")
                continue

            # PCA uygulama
            reduced_vectors, loss_percentage = apply_pca(combined_vectors)
            print(f"{vector_columns[0]} ile {col} birleşimi tamamlandı. Veri kaybı oranı: %{loss_percentage:.2f}")

            # Veritabanını güncelle
            serialized_vectors = [pickle.dumps(vec) for vec in reduced_vectors]
            update_database_column(conn, table_name, combined_col_name, serialized_vectors, df["id"])

            # PCA bilgilerini sakla
            combined_info.append({
                "VectorColumn": col,
                "LossPercentage": loss_percentage
            })

    except Exception as e:
        print(f"Hata: {e}")
    finally:
        conn.close()

    # PCA bilgilerini DataFrame olarak döndür
    info_df = pd.DataFrame(combined_info)
    if not info_df.empty:
        print("\nPCA ve Veri Kaybı Bilgileri:")
        print(info_df)
    else:
        print("\nHiçbir sütun için PCA bilgisi oluşturulamadı.")
    return info_df

# Ana işlem
db_path = "ElectionDb.db"
table_name = "elcPartC"
info_df = process_and_save_combined_vectors(db_path, table_name, VECTOR_COLUMNS)

# PCA bilgilerini CSV'ye kaydet
info_df.to_csv("T5_pca_info.csv", index=False)
print("İşlem tamamlandı ve PCA bilgileri kaydedildi.")

'''
"Cluster_Vector_DistilBERT","Cluster_Vector_RoBERTa",)
"Cluster_Vector_ALBERT","Cluster_Vector_T5",
"Cluster_Vector_sBERTVector_BERT", "Cluster_Vector_sBERTVector_DistilBERT","Cluster_Vector_sBERTVector_RoBERTa",
"Cluster_Vector_sBERTVector_ALBERT","Cluster_Vector_sBERTVector_T5",
"Cluster_Vector_sBERTVector_BERT","Cluster_Vector_T5Vector_BERT", "Cluster_Vector_T5Vector_DistilBERT",
"Cluster_Vector_T5Vector_RoBERTa", "Cluster_Vector_T5Vector_ALBERT",
"Cluster_Vector_T5Vector_sBERTVector_BERT", "Cluster_Vector_T5Vector_sBERTVector_DistilBERT",
"Cluster_Vector_T5Vector_sBERTVector_RoBERTa", "Cluster_Vector_T5Vector_sBERTVector_ALBERT"]
'''