import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt


'''
################################# Ülkelerin toplu şekilde küme dağılımını gösteriyor
# CSV dosyasını oku
df = pd.read_csv('topludeger.csv')

# Küme başlıklarını 0-7 için uygun şekilde ayarlama
df_columns = ['Country', 'Year', '0', '1', '2', '3', '4', '5', '6', '7']
df = df[df_columns]

# Küme değerlerindeki yüzde işaretlerini kaldırma ve sayıya çevirme
for küme in range(8):
    df[str(küme)] = df[str(küme)].str.rstrip('%').astype(float)  # Yüzdeyi sayıya çeviriyoruz
df = df[df['Year'] > 1999]
# Ülkeler ve yıllar için eksik olan verileri sıfır ile doldurma
all_years = sorted(df['Year'].unique())  # Tüm yıllar
all_countries = df['Country'].unique()  # Tüm ülkeler

# Her ülke ve yıl kombinasyonu için eksik olan veriyi sıfırla doldur
full_df = pd.DataFrame(columns=['Country', 'Year'] + [str(i) for i in range(8)])

# DataFrame'e her ülke için her yıl ekleyelim
for country in all_countries:
    for year in all_years:
        if not ((df['Country'] == country) & (df['Year'] == year)).any():
            new_row = {'Country': country, 'Year': year}
            new_row.update({str(i): 0.0 for i in range(8)})  # Küme 0-7 için sıfır ekle
            full_df = full_df._append(new_row, ignore_index=True)

# Eksik verileri sıfırla doldurduktan sonra, orijinal veriyi ekleyelim
df = pd.concat([df, full_df], ignore_index=True)

# Küme başlıklarını sıfırladıktan sonra, verileri tekrar sıralayalım
df = df.sort_values(by=['Country', 'Year'])
# Her küme için bir grafik oluştur
for küme in range(8):  # Küme 0'dan Küme 7'ye kadar
    plt.figure(figsize=(12, 8))

    for country in df['Country'].unique():
        df_country = df[df['Country'] == country]

        # Sıfır olmayan veriler için çizim yapıyoruz
        df_non_zero = df_country[df_country[str(küme)] > 0]

        # Küme için nokta çizimi yapalım
        plt.plot(df_non_zero['Year'],
                 df_non_zero[str(küme)],  # Sayısal değerlerle çiziyoruz
                 marker='o',  # Nokta kullanıyoruz
                 label=f'{country}',
                 linestyle='-',  # Çizgi olmasın, sadece noktalar olsun
                 markersize=8)  # Nokta boyutunu ayarlıyoruz

    # Grafik başlıkları ve gösterimi
    plt.title(f'Seçim Dönemi Ülkelerin Ağırlıklı Küme-{küme} Dağılım Oranı')
    plt.xlabel('Yıl')
    plt.ylabel('Yüzde')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.grid(True)

    # Grafik dosyasını kaydetme
    plt.tight_layout()
    plt.savefig(f'küme_{küme}.png')  # Her küme için bir PNG dosyası oluştur
    plt.close()  # Yeni grafiği oluşturmak için önceki grafiği kapatıyoruz
'''
pass


###### n gramlar üzerinden kelime bulutu çıkarıyor.
def clean_ngram(ngram):
    """2-gram temizleme fonksiyonu: Parantez ve tırnak işaretlerini kaldırır."""
    return ngram.replace("(", "").replace(")", "").replace("'", "").replace(",", " ").strip()
def generate_wordclouds_from_csv(directory, output_folder):
    # Tüm CSV dosyalarını listele
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Kümeler için kelime ve frekansları saklayacağımız sözlük
    cluster_data = {}

    # Tüm dosyaları oku ve verileri birleştir
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path)

        # Sütun adlarını kontrol et ve uygun hale getir
        df.columns = ['2-gram', 'Frequency', 'Cluster']

        # Her küme için verileri ekle
        for cluster in df['Cluster'].unique():
            if cluster not in cluster_data:
                cluster_data[cluster] = {}
            cluster_df = df[df['Cluster'] == cluster]
            for _, row in cluster_df.iterrows():
                ngram = clean_ngram(row['2-gram'])
                frequency = row['Frequency']
                if ngram in cluster_data[cluster]:
                    cluster_data[cluster][ngram] += frequency
                else:
                    cluster_data[cluster][ngram] = frequency
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for cluster, ngrams in cluster_data.items():
            wordcloud = WordCloud(width=800, height=600, background_color="white").generate_from_frequencies(ngrams)
            plt.figure(figsize=(10, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            #plt.title(f"Word Cloud for Cluster {cluster}", fontsize=16)

            output_path = os.path.join(output_folder, f"{csv_file}{cluster}_wordcloud.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Kelime bulutu kaydedildi: {output_path}")


# Kullanım
input_directory = "cloud"  # CSV dosyalarının bulunduğu klasör
output_directory = "cloud"  # Kelime bulutlarının kaydedileceği klasör
generate_wordclouds_from_csv(input_directory, output_directory)
