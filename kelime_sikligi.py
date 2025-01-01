import sqlite3
import pandas as pd
import sqlite3
import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
conn = sqlite3.connect('ElectionDb.db')
cursor = conn.cursor()
query = "SELECT Cluster_Vector, Corpus FROM elcPart WHERE Cluster_Vector IS NOT NULL"
cursor.execute(query)
data = cursor.fetchall()
df = pd.DataFrame(data, columns=["Cluster_Vector", "Corpus"])

conn.close()


def get_bigrams(text):
    # Küçük harfe çevirme ve özel karakterleri temizleme
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)  # Kelimeleri ayırma
    filtered_words = [word for word in words if word not in stop_words]  # Stop words'ları çıkarma

    # Bigramları oluşturma (yan yana iki kelime)
    bigrams = [(filtered_words[i], filtered_words[i + 1]) for i in range(len(filtered_words) - 1)]
    return bigrams


# Küme numarasına göre bigram frekanslarını hesaplama
bigram_frequencies = {}

for cluster_id, group in df.groupby('Cluster_Vector'):
    bigram_frequencies[cluster_id] = Counter()

    # Her metin için bigram frekanslarını toplama
    for text in group['Corpus']:
        bigrams = get_bigrams(text)
        bigram_frequencies[cluster_id] += Counter(bigrams)

# Sonuçları yazdırma (ilk 2 en fazla geçen bigramı)
for cluster_id, freq in bigram_frequencies.items():
    print(f"Cluster {cluster_id}:")
    for bigram, count in freq.most_common(100):  # İlk 2 en fazla geçen bigramı yazdırma
        print(f"  {bigram[0]} {bigram[1]}: {count}")

