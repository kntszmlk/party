import sqlite3
import pandas as pd
import re

# 1. SQLite veritabanına bağlan ve orjelectiontext tablosunu dataframe'e aktar
conn = sqlite3.connect("ElectionDb.db")
query = "SELECT id, CountryName, PartyCode, DateTime, Corpus FROM OrjElectionText"
df = pd.read_sql_query(query, conn)

# 2. partyNames.txt dosyasını okuyup bir diziye aktar
with open("partyNames.txt", "r", encoding="utf-8") as file:
    party_names = [line.strip().lower() for line in file]

# 3. Metin işleme fonksiyonu
def clean_and_replace(text, party_names):
    # Tüm karakterleri küçük harfe çevir
    text = text.lower()

    # Tab karakterlerini ve birden fazla boşluğu tek bir boşluğa indir
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Yabancı karakterleri temizle
    text = re.sub(r"[^a-z0-9\s,.]", "", text)

    # Tarih formatlarını {{date}} ile değiştir
    text = re.sub(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b", "{{date}}", text)

    # partyNames içerisindeki ifadeleri {{party}} ile değiştir
    for party in party_names:
        text = re.sub(rf"\b{re.escape(party)}\b", "{{party}}", text)

    return text

# clsCorpus sütununu oluştur ve metinleri işleyerek ekle
df["clsCorpus"] = df["Corpus"].apply(lambda x: clean_and_replace(x, party_names))

# 5. elcPart tablosunu oluştur
def create_elc_part_table():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS elcPart (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        CountryName TEXT,
        PartyCode TEXT,
        DateTime TEXT,
        Corpus TEXT
    )""")
    conn.commit()

create_elc_part_table()

# 6. clsCorpus sütunundaki her satırı al, 3 cümlelik parçalara böl ve yeni bir DataFrame'de topla
def split_into_chunks(text, chunk_size=3):
    sentences = text.split(". ")  # Nokta ile cümleleri ayır
    return [". ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

chunked_data = []
for index, row in df.iterrows():
    chunks = split_into_chunks(row["clsCorpus"])
    for chunk in chunks:
        chunked_data.append({
            "CountryName": row["CountryName"],
            "PartyCode": row["PartyCode"],
            "DateTime": row["DateTime"],
            "Corpus": chunk
        })

chunked_df = pd.DataFrame(chunked_data)

# 7. chunked_df'deki satırları elcPart tablosuna ekle
for index, row in chunked_df.iterrows():
    conn.execute(
        "INSERT INTO elcPart (CountryName, PartyCode, DateTime, Corpus) VALUES (?, ?, ?, ?)",
        (row["CountryName"], row["PartyCode"], row["DateTime"], row["Corpus"])
    )

conn.commit()

# İşlemlerin bittiğini kontrol etmek için ilk birkaç satırı kontrol edelim
print("İlk birkaç satır:\n", chunked_df.head())
conn.close()

