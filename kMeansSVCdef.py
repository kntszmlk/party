import pickle
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import  confusion_matrix
import os
import pandas as pd
import numpy as np
import sqlite3
from nltk.util import bigrams
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from PIL import Image
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def plot_confusion_matrix(y_true, col, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{col} Confusion Matrix", fontsize=14)
    plt.xlabel("Predict", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()
    #plt.show()
    output = f"{col}_conf.png"
    plt.savefig(output, dpi=600, bbox_inches="tight")
    return cm

def save_or_load_svc_model(model, col, X_train=None, y_train=None):
    pickle_file = f"SVC_{col}.pkl"
    if os.path.exists(pickle_file):
        print(f"{pickle_file} bulundu. Model yükleniyor...")
        return joblib.load(pickle_file)
    elif X_train is not None and y_train is not None:
        print(f"{pickle_file} bulunamadı. Model eğitiliyor ve kaydediliyor...")
        model.fit(X_train, y_train)
        print("Model eğitildi.")
        joblib.dump(model, pickle_file)
        return model
    else:
        raise ValueError("Eğitim verisi verilmeden model yüklenemez veya eğitilemez.")

def check_and_create_clusters(db_path, table_name, df, col, n_clusters=None):
    cluster_col = f"Cluster_{col}"

    if cluster_col not in df.columns:
        print(f"{cluster_col} sütunu bulunamadı. Kümeler oluşturuluyor...")
        vectors = df[col].dropna().to_numpy()
        vectors = np.array([pickle.loads(vec) for vec in vectors])

        if n_clusters is None:
            n_clusters = determine_optimal_clusters(vectors, col)
            #n_clusters = 8
        print(f"{col} için Optimum Küme Sayısı: {n_clusters}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        df[cluster_col] = labels
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            if cluster_col not in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})")]:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {cluster_col} INTEGER")
            for idx, label in zip(df["id"], labels):
                cursor.execute(f"UPDATE {table_name} SET {cluster_col} = ? WHERE id = ?", (int(label), idx))
            conn.commit()
            print(f"{cluster_col} sütunu veritabanına başarıyla kaydedildi.")
        except Exception as e:
            print(f"Veritabanına kayıt sırasında hata oluştu: {e}")
        finally:
            conn.close()
    else:
        print(f"{cluster_col} sütunu zaten mevcut. Kümeleme atlanıyor.")
        labels = df[cluster_col].to_numpy()
        kmeans = None
    return kmeans, labels

def find_message_for_vector_id(df, vector_id):
    result = df[df["id"] == vector_id]["Corpus"]
    return result.iloc[0] if not result.empty else None
'''
def find_most_similar(new_vector, df, col, new_message, new_id, kmeans=None):
    try:
        new_vector = np.array(pickle.loads(new_vector)).reshape(1, -1)
    except Exception as e:
        print(f"Yeni vektör dönüşüm hatası: {e}")
        return []

    results = []

    if kmeans is None:
        if f"Cluster_{col}" not in df.columns:
            print(f"Veri çerçevesinde '{f'Cluster_{col}'}' sütunu mevcut değil.")
            return []
        cluster_labels = df[f"Cluster_{col}"].unique()
    else:
        cluster_labels = range(kmeans.n_clusters)

    for cluster_label in cluster_labels:
        # İlgili kümedeki verileri al
        cluster_data = df[df[f"Cluster_{col}"] == cluster_label]
        if cluster_data.empty:
            continue

        try:
            # Küme vektörlerini dönüştür
            cluster_vectors = []
            for v in cluster_data[col].values:
                if v is not None:
                    try:
                        vec = np.array(pickle.loads(v)).reshape(1, -1)
                        cluster_vectors.append(vec)
                    except Exception as e:
                        print(f"Vektör işlenemedi: {e}")

            cluster_vectors = np.vstack(cluster_vectors)
            # Cosine benzerlik hesaplama yap
            similarities = cosine_similarity(new_vector, cluster_vectors).flatten()
            most_similar_idx = np.argmax(similarities)
            results.append({
                "Cluster": cluster_label,
                "VectorID": cluster_data["id"].iloc[most_similar_idx],
                "SimilarityScore": similarities[most_similar_idx]
            })

        except Exception as e:
            print(f"Küme işleme hatası (Cluster {cluster_label}): {e}")
            continue

    print(f"Mesaj: {new_message}")
    print(f"Vektör ID: {new_id}")
    for result in sorted(results, key=lambda x: x["SimilarityScore"], reverse=True):
        # id nin ait olduğu mesaj getiriliyor => find_message_for_vector_id
        similar_message = find_message_for_vector_id(df, result["VectorID"])
        print(f"Küme: {result['Cluster']}")
        print(f"Benzer Vektör ID: {result['VectorID']}")
        print(f"Benzerlik Skoru: {result['SimilarityScore']:.4f}")
        print(f"Benzer Mesaj: {similar_message}")
        print(f"---------------------------------------------")
'''
def find_most_similar(new_vector, df, col, new_message, new_id, kmeans=None):
    try:
        new_vector = np.array(pickle.loads(new_vector)).reshape(1, -1)  # 2D yapıyoruz
    except Exception as e:
        print(f"Yeni vektör dönüşüm hatası: {e}")
        return []

    results = []

    if kmeans is None:
        if f"Cluster_{col}" not in df.columns:
            print(f"Veri çerçevesinde '{f'Cluster_{col}'}' sütunu mevcut değil.")
            return []
        cluster_labels = df[f"Cluster_{col}"].unique()
    else:
        cluster_labels = range(kmeans.n_clusters)

    for cluster_label in cluster_labels:
        # İlgili kümedeki verileri al
        cluster_data = df[df[f"Cluster_{col}"] == cluster_label]
        if cluster_data.empty:
            continue

        try:
            # Küme vektörlerini dönüştür
            cluster_vectors = []
            for v in cluster_data[col].values:
                if v is not None:
                    try:
                        vec = np.array(pickle.loads(v)).reshape(1, -1)  # 2D hale getir
                        cluster_vectors.append(vec)
                    except Exception as e:
                        print(f"Vektör işlenemedi: {e}")

            # Tüm küme vektörlerini birleştir
            if cluster_vectors:
                cluster_vectors = np.vstack(cluster_vectors)
                # Cosine benzerlik hesaplama yap
                similarities = cosine_similarity(new_vector, cluster_vectors).flatten()
                most_similar_idx = np.argmax(similarities)
                results.append({
                    "Cluster": cluster_label,
                    "VectorID": cluster_data["id"].iloc[most_similar_idx],
                    "SimilarityScore": similarities[most_similar_idx]
                })
            else:
                print(f"Küme {cluster_label} için geçerli vektör bulunamadı.")

        except Exception as e:
            print(f"Küme işleme hatası (Cluster {cluster_label}): {e}")
            continue

    print(f"Mesaj: {new_message}")
    print(f"Vektör ID: {new_id}")
    for result in sorted(results, key=lambda x: x["SimilarityScore"], reverse=True):
        # id'nin ait olduğu mesajı getiriyoruz
        similar_message = find_message_for_vector_id(df, result["VectorID"])
        print(f"Küme: {result['Cluster']}")
        print(f"Benzer Vektör ID: {result['VectorID']}")
        print(f"Benzerlik Skoru: {result['SimilarityScore']:.4f}")
        print(f"Benzer Mesaj: {similar_message}")
        print(f"---------------------------------------------")


def determine_optimal_clusters(data, col, max_clusters=20, random_state=42):
    distortions = []
    K = range(1, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    deltas = np.diff(distortions)
    delta_ratios = np.abs(deltas[1:] / deltas[:-1])
    optimal_k = np.argmax(delta_ratios) + 1
    if len(delta_ratios) == 0:
        optimal_k = 3
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, marker='o', linestyle='-', color='b')
    plt.title("Elbow Yöntemi ile Optimum Küme Sayısı", fontsize=14)
    plt.xlabel("Küme Sayısı (k)", fontsize=12)
    plt.ylabel("Distortion (Inertia)", fontsize=12)
    plt.xticks(K)
    plt.axvline(optimal_k, color='r', linestyle='--', label=f"Optimum k = {optimal_k}")
    plt.legend()
    plt.grid(True)
    #plt.show()
    output = f"{col}_elbow.png"
    plt.savefig(output, dpi=600, bbox_inches="tight")
    print(f"Otomatik olarak belirlenen optimum küme sayısı: {optimal_k}")
    return optimal_k

def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    try:
        query = f"SELECT * FROM {table_name} "
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df

def find_cluster_for_new_vector(new_vector, df, col, svc_model=None):
    try:
        # Eğer new_vector pickle ile serileştirilmişse, onu çözümle
        if isinstance(new_vector, bytes):
            new_vector = np.array(pickle.loads(new_vector)).reshape(1, -1)
        elif isinstance(new_vector, np.ndarray):
            # Eğer new_vector zaten numpy array ise, doğrudan kullanabiliriz
            new_vector = new_vector.reshape(1, -1)
        else:
            raise ValueError("Beklenmeyen vektör türü, numpy array veya pickle formatında olmalı.")
    except Exception as e:
        print(f"Yeni vektör dönüşüm hatası: {e}")
        return {"PredictedClass": None}

    # Eğer SVC modeli verilmişse, tahmin yapıyoruz
    if svc_model is not None:
        try:
            predicted_class = svc_model.predict(new_vector)[0]  # modelin tahminini al
            print(f"Yeni vektörün tahmin edilen sınıfı (SVC): {predicted_class}")
        except Exception as e:
            print(f"SVC model tahmin hatası: {e}")
            predicted_class = None
    else:
        print("SVC modeli verilmedi.")
        predicted_class = None

    return {"PredictedClass": predicted_class}


def plot_clusters_pca_2d(cluster_labels, vectors, title_prefix):
    pca = PCA(n_components=2)
    cluster_vectors = vectors
    ''' 
    for v in vectors:
        if v is not None:
            try:
                vec = np.array(pickle.loads(v)).reshape(1, -1)  # 2D hale getir
                cluster_vectors.append(vec)
            except Exception as e:
                print(f"Vektör işlenemedi: {e}")
    '''
    pca_result = pca.fit_transform(cluster_vectors)
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    for i, label in enumerate(unique_labels):
        cluster_data = pca_result[cluster_labels == label]
        plt.scatter(
            cluster_data[:, 0], cluster_data[:, 1],
            label=f'Cluster {label}',
            color=colors(i),
            alpha=0.7
        )
    plt.title(f"{title_prefix} - PCA 2D")
    plt.xlabel("PCA - 1")
    plt.ylabel("PCA - 2")
    plt.legend(title="Cluster Numbers")
    plt.tight_layout()

    output = f"{title_prefix}_cluster.png"
    plt.savefig(output, dpi=600, bbox_inches="tight")
    # plt.show()
def plot_models_pca_2d(cluster_labels, vectors, title_prefix):
    pca = PCA(n_components=2)
    cluster_vectors = []
    for v in vectors:
        if v is not None:
            try:
                vec = np.array(pickle.loads(v)).reshape(1, -1)  # 2D hale getir
                cluster_vectors.append(vec)
            except Exception as e:
                print(f"Vektör işlenemedi: {e}")
    pca_result = pca.fit_transform(cluster_vectors)
    plt.figure(figsize=(4, 3))
    cluster_data = pca_result
    plt.scatter(
            cluster_data[:, 0], cluster_data[:, 1],
            color="black",
            alpha=0.7
    )
    #plt.title(f"{title_prefix} - PCA 2D")
    #plt.xlabel("PCA - 1")
    #plt.ylabel("PCA - 2")
    plt.tight_layout()

    output = f"{title_prefix}.png"
    plt.savefig(output, dpi=600, bbox_inches="tight")
    # plt.show()


def calculate_ngrams_for_clusters(df, text_col, cluster_col, output_csv_path_2gram, output_csv_path_1gram):

    stop_words = set(stopwords.words('english'))
    all_ngrams_df = pd.DataFrame()
    all_1grams_df = pd.DataFrame()

    # Her küme için iterasyon
    cluster_labels = df[cluster_col].unique()
    for cluster_label in cluster_labels:
        cluster_data = df[df[cluster_col] == cluster_label]
        cluster_ngrams = Counter()
        cluster_1grams = Counter()

        for text in cluster_data[text_col].dropna():
            tokens = nltk.word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            bigram_tokens = bigrams(tokens)

            # 2-gram frekansları
            cluster_ngrams.update(bigram_tokens)

            # 1-gram frekansları
            cluster_1grams.update(tokens)

        # Kümenin 2-gramlarını DataFrame'e ekle
        cluster_2gram_df = pd.DataFrame(cluster_ngrams.items(), columns=['2-gram', 'Frequency'])
        cluster_2gram_df['Cluster'] = cluster_label
        all_ngrams_df = pd.concat([all_ngrams_df, cluster_2gram_df], ignore_index=True)

        # Kümenin 1-gramlarını DataFrame'e ekle
        cluster_1gram_df = pd.DataFrame(cluster_1grams.items(), columns=['1-gram', 'Frequency'])
        cluster_1gram_df['Cluster'] = cluster_label
        all_1grams_df = pd.concat([all_1grams_df, cluster_1gram_df], ignore_index=True)

    # Frequency sütununu sayısal formata dönüştür
    all_ngrams_df['Frequency'] = pd.to_numeric(all_ngrams_df['Frequency'], errors='coerce')
    all_1grams_df['Frequency'] = pd.to_numeric(all_1grams_df['Frequency'], errors='coerce')

    # Her küme için ilk 100 sonucu filtrele
    top_2grams_df = (
        all_ngrams_df.groupby('Cluster', group_keys=False)
        .apply(lambda x: x.nlargest(100, 'Frequency'))
    )

    top_1grams_df = (
        all_1grams_df.groupby('Cluster', group_keys=False)
        .apply(lambda x: x.nlargest(100, 'Frequency'))
    )

    # Sonuçları CSV'ye kaydet
    top_2grams_df.to_csv(output_csv_path_2gram, index=False)
    top_1grams_df.to_csv(output_csv_path_1gram, index=False)

    print(f"Tüm kümelerin ilk 100 2-gram frekansları '{output_csv_path_2gram}' dosyasına kaydedildi.")
    print(f"Tüm kümelerin ilk 100 1-gram frekansları '{output_csv_path_1gram}' dosyasına kaydedildi.")

    # Kelime bulutlarını oluştur ve görselleştir
    generate_wordclouds(top_2grams_df, top_1grams_df, cluster_col)


def generate_wordclouds(top_2grams_df, top_1grams_df, cluster_col):
    cluster_labels = top_2grams_df['Cluster'].unique()

    for cluster_label in cluster_labels:
        # 2-gram kelime bulutu
        cluster_2gram_data = top_2grams_df[top_2grams_df['Cluster'] == cluster_label]
        word_freq_2gram = {
            ' '.join(ngram): freq
            for ngram, freq in zip(cluster_2gram_data['2-gram'], cluster_2gram_data['Frequency'])
        }

        wordcloud_2gram = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(word_freq_2gram)

        # 1-gram kelime bulutu
        cluster_1gram_data = top_1grams_df[top_1grams_df['Cluster'] == cluster_label]
        word_freq_1gram = {
            ' '.join(ngram): freq
            for ngram, freq in zip(cluster_1gram_data['1-gram'], cluster_1gram_data['Frequency'])
        }

        wordcloud_1gram = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(word_freq_1gram)

        # Görselleştir
        #plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_2gram, interpolation='bilinear')
        plt.axis('off')
        #plt.title(f'Cluster {cluster_label} 2-Gram Kelime Bulutu', fontsize=16)
        #plt.show()
        output = f"Cloud2_{cluster_col}_{cluster_label}.png"
        plt.savefig(output, dpi=600, bbox_inches="tight")

        #plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_1gram, interpolation='bilinear')
        plt.axis('off')
        #plt.title(f'{cluster_col}_{cluster_label} 1-Gram Kelime Bulutu', fontsize=16)
        #plt.show()
        output = f"Cloud1_{cluster_col}_{cluster_label}.png"
        plt.savefig(output, dpi=600, bbox_inches="tight")


def plot_clusters_pca_triple(cluster_labels, vectors, title, ax):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        cluster_data = pca_result[cluster_labels == label]
        ax.scatter(
            cluster_data[:, 0], cluster_data[:, 1],
            label=f'Cluster {label}',
            color=colors(i),
            alpha=0.7
        )
    ax.set_title(title)
    ax.set_xlabel("PCA - 1")
    ax.set_ylabel("PCA - 2")
    ax.legend(loc='best', title="Clusters")


def combine_images(image_paths, output_path, orientation='horizontal'):
    """
    Birden fazla resmi birleştirip tek bir PNG dosyası oluşturur.

    Args:
        image_paths (list): Birleştirilecek resim dosyalarının yolları.
        output_path (str): Oluşturulacak birleşik resmin kaydedileceği yol.
        orientation (str): 'horizontal' (yan yana) veya 'vertical' (alt alta) düzenleme.
    """
    images = [Image.open(img_path) for img_path in image_paths]

    # Resimlerin boyutlarını kontrol et
    widths, heights = zip(*(img.size for img in images))

    if orientation == 'horizontal':
        total_width = sum(widths)
        max_height = max(heights)
        combined_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]
    else:  # 'vertical'
        total_height = sum(heights)
        max_width = max(widths)
        combined_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.size[1]

    combined_image.save(output_path)
    print(f"Birleştirilmiş resim kaydedildi: {output_path}")
