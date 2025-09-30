from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Mapping Kecamatan ke Desa (Kelurahan)
kecamatan_desa_map = {
    "Asem Rowo": ["Asem Rowo", "Genting Kalianak", "Greges", "Tambak Sarioso"],
    "Benowo": ["Tambak Osowilangun", "Romokalisari", "Klakah Rejo", "Sememi", "Kandangan"],
    "Bubutan": ["Jepara", "Gundih", "Tembok Dukuh", "Alun-Alun Contong", "Bubutan"],
    "Bulak": [ "Sukolilo Baru", "Sukolilo Lor", "Kenjeran", "Bulak", "Kedung Cowek"],
    "Dukuh Pakis": ["Gunung Sari", "Dukuh Kupang", "Dukuh Pakis", "Pradah Kali Kendal"],
    "Gayungan": ["Ketintang", "Dukuh Menanggal", "Menanggal", "Gayungan"],
    "Genteng": ["Embong Kaliasin", "Ketabang", "Kapasari", "Peneleh", "Genteng"],
    "Gubeng": ["Gubeng", "Kertajaya", "Pucang Sewu", "Baratajaya", "Mojo", "Airlangga"],
    "Gunung Anyar": ["Rungkut Menanggal", "Rungkut Tengah", "Gunung Anyar", "Gunung Anyar Tambak"],
    "Jambangan": ["Jambangan", "Karah", "Kebonsari", "Pagesangan"],
    "Karangpilang": ["Karangpilang", "Waru Gunung", "Kebraon", "Kedurus"],
    "Kenjeran": ["Tambak Wedi", "Bulak Banteng", "Sidotopo Wetan", "Tanah Kali Kedinding"],
    "Krembangan": ["Krembangan Selatan", "Kemayoran", "Perak Barat", "Morokrembangan", "Dupak"],
    "Lakarsantri": ["Lakarsantri", "Jeruk", "Lidah Kulon", "Lidah Wetan", "Bangkingan", "Sumur Welut"],
    "Mulyorejo": ["Kalisari", "Kejawan Putih Tambak", "Dukuh Sutorejo", "Kalijudan", "Mulyorejo", "Manyar Sabrangan"],
    "Pabean Cantian": ["Bongkaran", "Nyamplungan", "Krembangan Utara", "Perak Timur", "Perak Utara", "Tanjung Perak"],
    "Pakal": ["Sumber Rejo", "Tambak Dono", "Benowo", "Pakal", "Babat Jerawat"],
    "Rungkut": ["Kali Rungkut", "Rungkut Kidul", "Medokan Ayu", "Wonorejo", "Penjaringansari", "Kedung Baruk"],
    "Sambikerep": ["Lontar", "Sambikerep", "Bringin", "Made"],
    "Sawahan": ["Sawahan", "Petemon", "Kupang Krajan", "Banyu Urip", "Putat Jaya", "Pakis"],
    "Semampir": ["Ampel", "Sidotopo", "Pegirian", "Wonokusumo", "Ujung"],
    "Simokerto": ["Kapasan", "Tambak Rejo", "Simokerto", "Simolawang", "Sidodadi"],
    "Sukolilo": ["Keputih", "Gebang Putih", "Klampis Ngasem", "Menur Pumpungan", "Nginden Jangkungan", "Medokan Semampir", "Semolowaru"],
    "Sukomanunggal": ["Tanjungsari", "Sukomanunggal", "Putat Gede", "Sonokwijenan", "Simomulyo", "Simomulyo Baru"],
    "Tambaksari": ["Pacarkeling", "Dukuh Setro", "Pacarkembang", "Ploso", "Gading", "Rangkah", "Kapas Madya", "Tambaksari"],
    "Tandes": ["Buntaran", "Banjar Sugihan", "Manukan Kulon", "Manukan Wetan", "Balong Sari", "Bibis", "Gedang Asin", "Karang Poh", "Tandes Kidul", "Tandes Lor", "Gadel", "Tubanan"],
    "Tegalsari": ["Kedungdoro", "Tegalsari", "Wonorejo", "Dr. Soetomo", "Keputran"],
    "Tenggilis Mejoyo": ["Kutisari", "Kendangsari", "Tenggilis Mejoyo", "Panjang Jiwo", "Prapen"],
    "Wiyung": ["Balas Klumprik", "Babatan", "Wiyung", "Jajar Tunggal"],
    "Wonocolo": ["Siwalankerto", "Jemur Wonosari", "Margorejo", "Bendul Merisi", "Sidosermo"],
    "Wonokromo": ["Darmo", "Sawunggaling", "Wonokromo", "Jagir", "Ngagelrejo", "Ngagel"]
}


# === Load model dan semua file pendukung ===
features_used = joblib.load("model/features_used_skenario11.pkl")
scaler = joblib.load("model/scaler_standard_skenario11.pkl")
model = load_model("model/best_model_11_32_64_32_seed118.h5")
kecamatan_mean_logharga = joblib.load("model/kecamatan_mean_logharga.pkl")
desa_mean_logharga = joblib.load("model/desa_mean_logharga.pkl")


# === Mapping kategori ===
kepemilikan_mapping = {'SHM': 2, 'HGB': 1, 'Girik': 0}
sumber_air_options = ['Air_PAM', 'Air_PDAM', 'Air_sumber air air tanah']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    harga_rp = None

    if request.method == 'POST':
        # Ambil input dari form
        form_data = request.form.to_dict()

        # Konversi tipe data numerik untuk fitur yang sesuai
        data = {}
        for feat in features_used:
            if feat in ['Kamar', 'Luas Tanah', 'Luas Bangunan', 'Kamar mandi', 'Lantai']:
                data[feat] = float(form_data.get(feat, 0))
            elif feat == 'Listrik':
                try:
                    listrik_val = float(form_data.get(feat, 0))
                    # Normalisasi sesuai data training (contoh: watt dibagi 1000)
                    data[feat] = listrik_val
                except ValueError:
                    data[feat] = 0
            elif feat in ['Ac', 'Akses 24/7', 'Akses mobil', 'Atm center', 'Balkon', 'Bathtub', 
                        'Carport', 'Cctv', 'Garasi', 'Jogging track', 'Kasur',
                        'Keamanan 24 jam', 'Kitchen set', 'Lemari pakaian', 'Pompa air', 
                        'Shower', 'Tangki air', 'Water heater']:
                data[feat] = 1 if form_data.get(feat) == 'on' else 0

        # === Encoding kecamatan & desa ===
        kecamatan = form_data.get('Kecamatan', '')
        desa = form_data.get('Desa', '')
        data['enc_kecamatan'] = kecamatan_mean_logharga.get(kecamatan, np.mean(list(kecamatan_mean_logharga.values())))
        data['enc_desa'] = desa_mean_logharga.get(desa, np.mean(list(desa_mean_logharga.values())))

        # === Encoding Kepemilikan Tanah ===
        kepemilikan = form_data.get('Kepemilikan Tanah', '')
        data['Kepemilikan Tanah_encoded'] = kepemilikan_mapping.get(kepemilikan, 0)

        # === Encoding Sumber Air ===
        sumber_air = form_data.get('Sumber Air', '')
        for opt in sumber_air_options:
            data[opt] = 1 if opt.split("_", 1)[-1].lower() in sumber_air.lower() else 0

        # === Susun DataFrame sesuai urutan fitur ===
        row = [data.get(feat, 0) for feat in features_used]
        X = pd.DataFrame([row], columns=features_used)

        # === Debug urutan fitur ===
        print("=== CEK URUTAN FITUR ===")
        print("Fitur saat training:")
        for i, f in enumerate(features_used):
            print(i, f)

        print("Fitur saat inference:", X.columns.tolist())
        print("Row sebelum scaling:", row)

        # === Scaling ===
        cols_to_scale = ["Luas Tanah", "Luas Bangunan", "Kamar", "Kamar mandi", "Listrik"]
        X_scaled = X.copy()
        X_scaled[cols_to_scale] = scaler.transform(X[cols_to_scale])

        print("Features_used")
        print(features_used)
        print("=== DATA mentah ===")
        print(data)
        print("=== Row sebelum scaling ===")
        print(row)
        print("=== Row setelah scaling ===")
        pd.set_option("display.max_columns", None)
        print(X_scaled.head())


        # === Prediksi ===
        log_pred = model.predict(X_scaled)[0][0]
        log_pred = min(log_pred, 24)
        prediction = round(log_pred, 8)
        harga_rp = round(np.exp(log_pred))

    return render_template('index.html',
                           prediction=prediction,
                           harga_rp="{:,}".format(harga_rp) if harga_rp else None,
                           kecamatan_list=list(kecamatan_desa_map.keys()),
                           kecamatan_desa_map=kecamatan_desa_map,
                           form_data=form_data if request.method == 'POST' else None)

if __name__ == '__main__':
    app.run(debug=True)
