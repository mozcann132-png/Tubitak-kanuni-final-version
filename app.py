import streamlit as st
import pandas as pd
import datetime
import re
import os
from langchain_groq import ChatGroq

# --- 1. SAYFA AYARLARI VE TASARIM (CSS) ---
st.set_page_config(page_title="Kanuni Asistan", page_icon="🏫", layout="centered")

# Özel CSS ile Arayüzü Güzelleştirme
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #0052cc;
        text-align: center;
        margin-bottom: 10px;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    /* Sidebar stil */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    </style>
    <div class="main-title">🏫 Kanuni Öğrenci Asistanı</div>
    """, unsafe_allow_html=True)

# --- 2. YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluent/100/000000/school.png", width=80)
    st.title("Proje Bilgileri")
    st.info("Bu yapay zeka asistanı, Kanuni MTAL öğrencileri için TÜBİTAK projesi kapsamında geliştirilmiştir.")
    st.markdown("---")
    st.subheader("💡 İpuçları")
    st.write("- '9A yarın ilk ders ne?'")
    st.write("- '10B bugün 3. ders nerede?'")
    st.write("- 'Ders araları kaç dakika?'")
    

# --- 3. API VE MODEL KURULUMU ---
try:
    # Streamlit Secrets üzerinden API anahtarını al
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("⚠️ Hata: 'GROQ_API_KEY' bulunamadı. Lütfen Streamlit Cloud ayarlarından Secrets kısmına ekleyin!")
    st.stop()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0, 
    groq_api_key=groq_api_key
)

# --- 4. VERİ YÜKLEME ---
@st.cache_data
def veri_yukle():
    try:
        # CSV dosyanın GitHub'da aynı dizinde olduğundan emin ol
        return pd.read_csv('SinifProgrami1404.csv', sep=';')
    except FileNotFoundError:
        st.error("⚠️ 'SinifProgrami1404.csv' dosyası bulunamadı!")
        return None

df = veri_yukle()

# --- 5. ASİSTAN FONKSİYONU ---
def asistan_yaniti(soru):
    if df is None:
        return "Veri dosyası yüklenemediği için yardımcı olamıyorum."

    # Gün hesaplama
    gunler_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
    bugun_adi = gunler_tr[datetime.datetime.now().weekday()]
    yarin_adi = gunler_tr[(datetime.datetime.now().weekday() + 1) % 7]

    # NLP Adımı: Sorudan verileri ayıkla
    nlu_prompt = f"""
    Soru: "{soru}"
    Bugün: {bugun_adi}, Yarın: {yarin_adi}
    Aşağıdaki bilgileri bul ve formatta yaz:
    SINIF:[Örn: 9-A], GUN:[Örn: Pazartesi], SAAT:[Rakam veya HEPSİ]
    Yanıt Formatı: SINIF:[...], GUN:[...], SAAT:[...]
    """

    try:
        nlu_cikti = llm.invoke(nlu_prompt).content
        h_sinif = re.search(r"SINIF:\[?(.*?)\]?,", nlu_cikti).group(1).replace("-", "").strip()
        h_gun = re.search(r"GUN:\[?(.*?)\]?,", nlu_cikti).group(1).strip()
        h_saat = re.search(r"SAAT:\[?(.*?)\]?$", nlu_cikti).group(1).strip()

        # Pandas Filtreleme
        mask = (df['Sinif'].str.replace("-", "").str.contains(h_sinif, case=False, na=False)) & \
               (df['Gun'].str.contains(h_gun, case=False, na=False))

        if h_saat.isdigit():
            mask = mask & (df['Girilen Ders Saati'] == int(h_saat))

        sonuc_df = df[mask]
        context = sonuc_df.to_string(index=False) if not sonuc_df.empty else "Kayıt bulunamadı."

    except:
        return "Üzgünüm, sorunu tam anlayamadım. Lütfen sınıf ve gün belirterek tekrar sorar mısın?"

    # Final Yanıt Oluşturma
    final_prompt = f"""
    Sen Kanuni MTAL okul asistanısın. Aşağıdaki verileri kullanarak öğrenciye samimi bir cevap ver.
    Veri yoksa uydurma yapma.
    VERİ: {context}
    SORU: {soru}
    """
    return llm.invoke(final_prompt).content

# --- 6. SOHBET ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Selam! Ben Kanuni asistanı. Sana nasıl yardımcı olabilirim?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            response = asistan_yaniti(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
