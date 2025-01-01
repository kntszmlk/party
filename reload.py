import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# İlk veritabanına bağlan
conn1 = sqlite3.connect("ElectionDb.db")
query1 = "SELECT id, Vector FROM elcPart WHERE Vector IS NOT NULL LIMIT 10"
cursor1 = conn1.cursor()
cursor1.execute(query1)

# İlk veritabanındaki vektörleri çek ve bir sözlükte sakla
dizi0 = {}
for row in cursor1.fetchall():
    record_id = row[0]
    vector_blob = row[1]
    vector = pickle.loads(vector_blob)  # BLOB'dan vektörü çıkar
    dizi0[record_id] = vector
conn1.close()

# İkinci veritabanına bağlan
conn2 = sqlite3.connect("ElectionDb2.db")
query2 = "SELECT id, sbert_vector FROM elcPart WHERE sbert_vector IS NOT NULL LIMIT 3"
cursor2 = conn2.cursor()
cursor2.execute(query2)

# İkinci veritabanındaki vektörleri çek ve bir sözlükte sakla
dizi1 = {}
for row in cursor2.fetchall():
    record_id = row[0]
    vector = np.frombuffer(row[1], dtype=np.float64)  # BLOB'dan vektörü çıkar
    dizi1[record_id] = vector
conn2.close()

# Vektörleri karşılaştır
for record_id in dizi0.keys():
    # Her iki vektörü de 2D şekline getir
    vector1_2d = dizi1[1].reshape(1, -1)  # Tek bir örnek için (1, n_features)
    vector2_2d = dizi0[record_id].reshape(1, -1)  # Tek bir örnek için (1, n_features)

    # Cosine similarity hesaplama
    similarity = cosine_similarity(vector1_2d, vector2_2d)
    print(f"Benzerlik oranı {record_id}: {similarity[0][0]:.4f}")

    # Eşitlik kontrolü (eşik değeri ile)
    similarity_threshold = 0.99  # 0.99 gibi bir eşik belirleyebiliriz
    if similarity[0][0] >= similarity_threshold:
        print(f"Kayıt {record_id}: Model tutarlı, benzerlik {similarity[0][0]:.4f}.")
    else:
        print(f"Kayıt {record_id}: Model tutarsız, benzerlik {similarity[0][0]:.4f}.")
