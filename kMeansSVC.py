from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, davies_bouldin_score, \
    calinski_harabasz_score
from kMeansSVCdef import *

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

columns = ["VectorColumn", "SilhouetteScore", "DaviesBouldinIndex", "CalinskiHarabaszIndex", "Accuracy"]
results_df = pd.DataFrame(columns=columns)

#vector_columns = ["Vector_sBERTVector_BERT", "Vector_sBERTVector_DistilBERT","Vector_sBERTVector_RoBERTa","Vector_sBERTVector_ALBERT","Vector_sBERTVector_T5"]
#vector_columns = ["Vector_sBERTVector_BERT","Vector_T5Vector_BERT", "Vector_T5Vector_DistilBERT", "Vector_T5Vector_RoBERTa", "Vector_T5Vector_ALBERT","Vector_T5Vector_sBERTVector_BERT", "Vector_T5Vector_sBERTVector_DistilBERT", "Vector_T5Vector_sBERTVector_RoBERTa", "Vector_T5Vector_sBERTVector_ALBERT"]
#vector_columns0 =["Vector_BERT","Vector_sBERT","Vector_DistilBERT", "Vector_RoBERTa", "Vector_ALBERT","Vector_T5"]
#vector_columns1 =["Vector_sBERTVector_BERT", "Vector_sBERTVector_DistilBERT","Vector_sBERTVector_RoBERTa","Vector_sBERTVector_ALBERT","Vector_sBERTVector_T5"]
#vector_columns2 =["Vector_T5Vector_BERT", "Vector_T5Vector_DistilBERT", "Vector_T5Vector_RoBERTa", "Vector_T5Vector_ALBERT"]
#vector_columns =["Vector_T5Vector_sBERTVector_BERT", "Vector_T5Vector_sBERTVector_DistilBERT", "Vector_T5Vector_sBERTVector_RoBERTa", "Vector_T5Vector_sBERTVector_ALBERT"]
vector_columns =["Vector_BERT","Vector_sBERT","Vector_DistilBERT", "Vector_RoBERTa", "Vector_ALBERT","Vector_T5","Vector_sBERTVector_BERT", "Vector_sBERTVector_DistilBERT","Vector_sBERTVector_RoBERTa","Vector_sBERTVector_ALBERT","Vector_sBERTVector_T5","Vector_T5Vector_BERT", "Vector_T5Vector_DistilBERT", "Vector_T5Vector_RoBERTa", "Vector_T5Vector_ALBERT","Vector_T5Vector_sBERTVector_BERT", "Vector_T5Vector_sBERTVector_DistilBERT", "Vector_T5Vector_sBERTVector_RoBERTa", "Vector_T5Vector_sBERTVector_ALBERT"]

df = load_data("ElectionDb.db", "elcPartC")
#df_test = load_data("ElectionDb.db", "elcPartT")
results = []
output_data = []
n_clusters = 8
'''
####### Mevcut png leri birleştirme#######################
output_folder_conf = "combined_confusion_matrices"
output_folder_elbow = "combined_elbow_graphs"
os.makedirs(output_folder_conf, exist_ok=True)
os.makedirs(output_folder_elbow, exist_ok=True)

group_size = 2  # Kaçlı gruplar halinde birleştirileceği
for i in range(0, len(vector_columns), group_size):
    group_columns = vector_columns[i:i + group_size]

    # Confusion Matrices için
    conf_image_paths = [f"{col}_conf.png" for col in group_columns]
    output_conf_path = os.path.join(output_folder_conf, f"combined_conf_{i // group_size + 1}.png")
    combine_images(conf_image_paths, output_conf_path, orientation='horizontal')

    # Elbow Grafikleri için
    elbow_image_paths = [f"{col}_elbow.png" for col in group_columns]
    output_elbow_path = os.path.join(output_folder_elbow, f"combined_elbow_{i // group_size + 1}.png")
    combine_images(elbow_image_paths, output_elbow_path, orientation='horizontal')
'''
######################### küme çıktıları
'''
output_folder =""
group_size = 2
for i in range(0, len(vector_columns), group_size):
    group_columns = vector_columns[i:i + group_size]
    fig, axes = plt.subplots(1, len(group_columns), figsize=(15, 5))

    for j, col in enumerate(group_columns):
        vectors = df[col].dropna().to_numpy()
        vectors = np.array([pickle.loads(vec) for vec in vectors])
        cluster_column_name = f"Cluster_{col}"
        cluster_labels = df[cluster_column_name].dropna().to_numpy()

        plot_clusters_pca_triple(cluster_labels, vectors, col, axes[j])

    output_filename = os.path.join(output_folder, f"combined_clusters_{i // group_size + 1}.png")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=600, bbox_inches="tight")
    plt.close()
'''

for col in vector_columns:
    print(f"\n--- {col} Sütunu İçin İşlem Başlıyor ---")
    # İlgili modele ait kümeleme olup olmadığına bakılıyor. => check_and_create_clusters
    kmeans, labels = check_and_create_clusters("ElectionDb.db", "elcPartC", df, col)
    vectors = df[col].dropna().to_numpy()
    vectors = np.array([pickle.loads(vec) for vec in vectors])
    # Tahminlr için SVC model varsa yükleniyor yoksa oluşturuluyor-kaydediliyor. => save_or_load_svc_model
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)
    svc_model = SVC(kernel='rbf', random_state=42, probability=True)
    svc_model = save_or_load_svc_model(svc_model, col, X_train, y_train)
    y_pred = svc_model.predict(X_test)
    # Conf Matrixler oluşturuluyor => plot_confusion_matrix
    #conf_matrix = plot_confusion_matrix(y_test, col, y_pred, labels=sorted(set(labels)))
    cluster_similarities = {}

    '''
    for row in df_test.itertuples(index=False):
        try:
            #new_vector = np.frombuffer(getattr(row, col), dtype=np.float64)
            new_vector = np.array(pickle.loads(getattr(row, col)))
            # Yeni verinin hangi kümeye ait olduğu belirleniyor => find_cluster_for_new_vector
            result = find_cluster_for_new_vector(new_vector, df, col, svc_model=svc_model)
            # Tüm kümeler içerisinde Yeni veriye en benzer veri bulunuyor => find_most_similar
            find_most_similar(getattr(row, col), df, col, str(row.Corpus), row.id, kmeans)
        except Exception as e:
            print(f"Hata: {e}")
            continue
    '''

    silhouette = silhouette_score(vectors, labels)
    davies_bouldin = davies_bouldin_score(vectors, labels)
    calinski_harabasz = calinski_harabasz_score(vectors, labels)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{col} - Silhouette Skoru: {silhouette:.4f}")
    print(f"{col} - Davies-Bouldin Endeksi: {davies_bouldin:.4f}")
    print(f"{col} - Calinski-Harabasz Endeksi: {calinski_harabasz:.4f}")

    row = {
        "VectorColumn": col,
        "SilhouetteScore": silhouette,
        "DaviesBouldinIndex": davies_bouldin,
        "CalinskiHarabaszIndex": calinski_harabasz,
        "Accuracy": accuracy
    }
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

#df_output01 = pd.DataFrame(output_data)
#df_output01.to_csv("deneme_sonuclar.csv", index=False)
df_output = pd.DataFrame(results_df)
df_output.to_csv("clustering_results_all.csv", index=False)
