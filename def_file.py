from sentence_transformers import SentenceTransformer
import os

def model_download():
    models = [
        'bert-base-nli-mean-tokens',
        'distilbert-base-nli-stsb-mean-tokens',
        'albert-base-v2',
        'xlnet-base-cased',
        'roberta-base',
        't5-base'
    ]

    # Yerel dizin
    local_model_dir = './models'

    # Modelleri yerel olarak kaydetme
    for model_name in models:
        model_path = os.path.join(local_model_dir, model_name.replace('/', '_'))  # Modelin kaydedileceği yol

        # Eğer model zaten mevcutsa, atla
        if os.path.exists(model_path):
            print(f"{model_name} modeli zaten mevcut. Atlanıyor...")
            continue

        print(f"{model_name} modeli indiriliyor ve kaydediliyor...")
        model = SentenceTransformer(model_name)
        model.save(model_path)  # Yerel klasöre kaydet
        print(f"{model_name} modeli başarıyla kaydedildi!")
