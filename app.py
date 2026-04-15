import streamlit as st
import pandas as pd
import datetime
import re
import os
from langchain_groq import ChatGroq

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kanuni Asistan", page_icon="🏫", layout="centered")
st.title("🏫 Kanuni Öğrenci Asistanı")
st.caption("Ders programınla ilgili her şeyi bana sorabilirsin!")

# --- API VE MODEL KURULUMU ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("⚠️ Lütfen Streamlit ayarlarından (Secrets) GROQ_API_KEY bilgisini girin!")
    st.stop()

# LLM Tanımlaması
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0, 
    groq_api_key=groq_api_key
)

# --- VERİ YÜKLEME (Önbellekli) ---
@st.cache_data
def veri_yukle():
    try:
        return pd.read_csv('SinifProgrami1404.csv', sep=';')
    except FileNotFoundError:
        st.error("⚠️ 'SinifProgrami1404.csv' dosyası bulunamadı. Lütfen GitHub deponuzda olduğundan emin olun.")
        st.stop()

df = veri_yukle()

# --- ASİSTAN BEYNİ (Core Fonksiyon) ---
def ogrenci_asistani_kesin_cozum(soru):
    # ADIM 1: Gün Hesaplama
    gunler_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
    bugun_adi = gunler_tr[datetime.datetime.now().weekday()]
    yarin_adi = gunler_tr[(datetime.datetime.now().weekday() + 1) % 7]

    # ADIM 2: Bilgi Ayıklama (LLM)
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
        h_sinif = re.search(r"SINIF:\[?(.*?)\]?,", cikti).group(1).replace("-", "").strip()
        h_gun = re.search(r"GUN:\[?(.*?)\]?,", cikti).group(1).strip()
        h_saat = re.search(r"SAAT:\[?(.*?)\]?$", cikti).group(1).strip()

        # ADIM 3: Pandas Filtreleme
        mask = (df['Sinif'].str.replace("-", "").str.contains(h_sinif, case=False, na=False)) & \
               (df['Gun'].str.contains(h_gun, case=False, na=False))

        if h_saat.isdigit():
            mask = mask & (df['Girilen Ders Saati'] == int(h_saat))

        sonuc_df = df[mask]

        if sonuc_df.empty:
            context = "Kayıt bulunamadı."
        else:
            context = sonuc_df.to_string(index=False)

    except Exception as e:
        return "Selam! Sınıfını veya hangi günü sorduğunu tam anlayamadım. (Örn: 9A pazartesi 3. ders ne?) 😊"

    # ADIM 4: Öğrenci Dostu Yanıt (LLM)
    final_prompt = f"""
    Sen çok yardımsever bir okul asistanısın.
    Verilen tabloyu kullanarak öğrenciye net, kısa ve samimi bir cevap ver.
    Cevabını verirken asla uydurma yapma, sadece tablodaki ders ismini, hocayı ve yeri söyle.

    TABLO VERİSİ:
    {context}

    ÖĞRENCİ SORUSU: {soru}
    """
    
    return llm.invoke(final_prompt).content

# --- STREAMLİT SOHBET ARAYÜZÜ ---

# Geçmiş mesajları sakla
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben Kanuni asistanıyım. Bana sınıfını ve gününü söyleyerek ders programını sorabilirsin. (Örn: 10-A yarın ilk dersim nerede?)"}
    ]

# Geçmiş mesajları ekrana çiz
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni girdi al
if prompt := st.chat_input("Ders programını sor..."):
    # Kullanıcı mesajını ekrana ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistanın düşünme efekti ve cevabı
    with st.chat_message("assistant"):
        with st.spinner("Program kontrol ediliyor..."):
            cevap = ogrenci_asistani_kesin_cozum(prompt)
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
