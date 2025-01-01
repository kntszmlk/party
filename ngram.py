from kMeansSVCdef import *

vector_columns = ["Cluster_Vector_T5Vector_BERT"]

df = load_data("ElectionDb.db", "elcPartC")

for col in vector_columns:
    if col in df.columns:  # Sütunun varlığını kontrol et
        output_csv_path_1gram = f"{col}_1grams.csv"
        output_csv_path_2gram = f"{col}_2grams.csv"
        result_df = calculate_ngrams_for_clusters(
            df=df,  # DataFrame
            text_col="Corpus",  # Metin sütunu adı
            cluster_col=col,  # Küme sütunu adı
            output_csv_path_2gram=output_csv_path_2gram,
            output_csv_path_1gram=output_csv_path_1gram # Çıkış dosyası
        )
    else:
        print(f"Sütun {col} DataFrame'de mevcut değil!")

