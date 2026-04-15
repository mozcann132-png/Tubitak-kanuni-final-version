import pandas as pd
import os
import datetime
import re
import streamlit as st
from langchain_groq import ChatGroq

# 1. AYARLAR & GÜVENLİK
# Streamlit secrets üzerinden API anahtarını alıyoruz
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Lütfen Streamlit Cloud ayarlarından GROQ_API_KEY bilgisini girin!")

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# Veri setini yükle (Dosya isminin aynı olduğundan emin ol)
@st.cache_data
def veri_yukle():
    return pd.read_csv('SinifProgrami1404.csv', sep=';')

df = veri_yukle()

# 2. ASİSTAN FONKSİYONU
def ogrenci_asistani_kesin_cozum(soru):
    gunler_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
    bugun_adi = gunler_tr[datetime.datetime.now().weekday()]
    yarin_adi = gunler_tr[(datetime.datetime.now().weekday() + 1) % 7]

    nlu_prompt = f"""
    Soru: "{soru}"
    Bugün: {bugun_adi}, Yarın: {yarin_adi}

    Sorudaki şu bilgileri bul ve sadece istenen formatta yaz:
    1. SINIF: (Örn: 9-A)
    2. GUN: (Pazartesi, Salı vb. Standart yaz)
    3. SAAT: (Sadece rakam yaz, Örn: 3. saat ise 3 yaz. Belirtilmediyse 'HEPSİ' yaz)

    Yanıt Formatı: SINIF:[...], GUN:[...], SAAT:[...]
    """

    try:
        cikti = llm.invoke(nlu_prompt).content
        h_sinif = re.search(r"SINIF:\[?(.*?)\],", cikti).group(1).replace("-", "").strip()
        h_gun = re.search(r"GUN:\[?(.*?)\],", cikti).group(1).strip()
        h_saat = re.search(r"SAAT:\[?(.*?)\s*$", cikti).group(1).replace("]", "").strip()

        mask = (df['Sinif'].str.replace("-", "").str.contains(h_sinif, case=False, na=False)) & \
               (df['Gun'].str.contains(h_gun, case=False, na=False))

        if h_saat.isdigit():
            mask = mask & (df['Girilen Ders Saati'] == int(h_saat))

        sonuc_df = df[mask]
        context = sonuc_df.to_string(index=False) if not sonuc_df.empty else "Kayıt bulunamadı."

    except Exception as e:
        return "Sınıfını veya gün bilgisini tam anlayamadım. Lütfen '10A yarın 3. ders ne?' gibi net sorar mısın?"

    final_prompt = f"""
    Sen yardımsever bir okul asistanısın. Sadece tablo verisine sadık kalarak cevap ver.
    TABLO VERİSİ: {context}
    ÖĞRENCİ SORUSU: {soru}
    """
    return llm.invoke(final_prompt).content

# 3. STREAMLIT ARAYÜZÜ
st.set_page_config(page_title="Okul Asistanı", page_icon="📚")
st.title("🏫 Kanuni Tübitak Okul Asistanı")
st.write("Sınıfını ve merak ettiğin dersi sorabilirsin.")

user_input = st.text_input("Sorunuzu buraya yazın:", placeholder="Örn: 10-A yarın 1. ders ne?")

if user_input:
    with st.spinner('Ders programı kontrol ediliyor...'):
        cevap = ogrenci_asistani_kesin_cozum(user_input)
        st.chat_message("assistant").write(cevap)
