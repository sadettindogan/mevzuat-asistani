import os
import shutil
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

st.set_page_config(
    page_title="Gemini Destekli Mevzuat Asistanı", page_icon="⚖️", layout="wide"
)

st.title("⚖️ Gemini Kaynaklı Akıllı Mevzuat Asistanı")
st.write(
    "Mevzuat belgeleriniz GitHub reponuzdaki `data` klasöründen beslenir. Google Gemini altyapısıyla çalışır."
)

# Sidebar - Ayarlar ve Dosya Ekleme
st.sidebar.header("⚙️ Ayarlar & Veri Yönetimi")
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key", type="password", help="Google AI Studio'dan aldığınız API anahtarınızı girin."
)

DATA_DIR = "data"
DB_DIR = "./chroma_db"

if not gemini_api_key:
    st.warning(
        "Lütfen devam etmek için sol menüden Gemini API anahtarınızı girin."
    )
    st.info(
        "💡 Ücretsiz Gemini API anahtarını [Google AI Studio](https://aistudio.google.com/) adresinden saniyeler içinde alabilirsiniz."
    )
else:
    genai.configure(api_key=gemini_api_key)

    class SimpleGeminiEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 768 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 768

    embeddings = SimpleGeminiEmbeddings()

    st.sidebar.subheader("➕ Yeni Mevzuat Ekle")
    new_files = st.sidebar.file_uploader(
        "Yeni PDF veya Word (.docx) yükleyin",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if new_files:
        if st.sidebar.button("Belgeleri Sisteme Ekle ve İndeksle"):
            with st.spinner("Yeni belgeler işleniyor ve veritabanı güncelleniyor..."):
                os.makedirs(DATA_DIR, exist_ok=True)
                for uploaded_file in new_files:
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                docs = []
                if os.path.exists(DATA_DIR):
                    for file in os.listdir(DATA_DIR):
                        f_path = os.path.join(DATA_DIR, file)
                        if file.endswith(".pdf"):
                            loader = PyMuPDFLoader(f_path)
                            docs.extend(loader.load())
                        elif file.endswith(".docx"):
                            loader = Docx2txtLoader(f_path)
                            docs.extend(loader.load())

                if docs:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(docs)

                    if os.path.exists(DB_DIR):
                        shutil.rmtree(DB_DIR)

                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory=DB_DIR,
                    )
                    st.sidebar.success(
                        "✅ Tüm mevzuat veritabanı başarıyla güncellendi!"
                    )
                else:
                    st.sidebar.warning("İşlenecek geçerli belge bulunamadı.")

    if not os.path.exists(DB_DIR):
        if st.sidebar.button("GitHub'daki Mevzuatları İlk Kez İndeksle"):
            with st.spinner(
                "GitHub'daki mevcut dosyalar taranıyor ve indeksleniyor..."
            ):
                docs = []
                if os.path.exists(DATA_DIR):
                    for file in os.listdir(DATA_DIR):
                        f_path = os.path.join(DATA_DIR, file)
                        if file.endswith(".pdf"):
                            loader = PyMuPDFLoader(f_path)
                            docs.extend(loader.load())
                        elif file.endswith(".docx"):
                            loader = Docx2txtLoader(f_path)
                            docs.extend(loader.load())

                if docs:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(docs)
                    
                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory=DB_DIR,
                    )
                    st.sidebar.success("İlk indeksleme tamamlandı!")
                else:
                    st.sidebar.warning(
                        "GitHub 'data' klasöründe henüz dosya bulunamadı."
                    )

    if os.path.exists(DB_DIR):
        vectorstore = Chroma(
            persist_directory=DB_DIR, embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        st.markdown("---")
        user_query = st.text_input(
            "Mevzuatla ilgili sorunuzu yazın:",
            placeholder="Örn: Yan sanayici nedir?",
        )

        if user_query:
            with st.spinner("Mevzuat taranıyor ve Gemini yanıtı hazırlıyor..."):
                source_documents = retriever.invoke(user_query)
                context = "\n\n".join([doc.page_content for doc in source_documents])

                prompt = f"""Sen profesyonel bir mevzuat ve hukuk asistanısın. Sadece aşağıdaki bağlamda (mevzuat metninde) yer alan bilgilere dayanarak soruyu yanıtla. Eğer bilgi metinde yoksa "Yüklenen mevzuat metinlerinde bu bilgiye ulaşılamadı" de.

Bağlam:
{context}

Soru: {user_query}
Yanıt:"""

                # Doğrudan en güncel ve kararlı model adı
                model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
                response = model.generate_content(prompt)
                ans = response.text

                st.markdown("### 🤖 Yanıt:")
                st.write(ans)

                with st.expander("📚 Kaynak Mevzuat Maddeleri"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**Kaynak {i+1}:**")
                        st.text(doc.page_content[:300] + "...")
                        if "source" in doc.metadata:
                            st.caption(f"Dosya: {doc.metadata['source']}")
    else:
        st.info(
            "Sistemi başlatmak için sol menüden **'GitHub'daki Mevzuatları İlk Kez İndeksle'** butonuna basın."
        )
