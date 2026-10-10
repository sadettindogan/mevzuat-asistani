import os
import shutil
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="GitHub Destekli Mevzuat Asistanı", page_icon="⚖️", layout="wide"
)

st.title("⚖️ GitHub Kaynaklı Akıllı Mevzuat Asistanı")
st.write(
    "Mevzuat belgeleriniz GitHub reponuzdaki `data` klasöründen beslenir. Yeni eklenen belgeleri sisteme dahil etmek için güncelleyebilirsiniz."
)

# Sidebar - Ayarlar ve Dosya Ekleme
st.sidebar.header("⚙️ Ayarlar & Veri Yönetimi")
api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="OpenAI API anahtarınızı girin."
)

DATA_DIR = "data"
DB_DIR = "./chroma_db"

if not api_key:
    st.warning(
        "Lütfen devam etmek için sol menüden OpenAI API anahtarınızı girin."
    )
else:
    os.environ["OPENAI_API_KEY"] = api_key

    # Zamanla yeni mevzuat eklemek için arayüzden yükleme alanı
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

                # Tüm data klasörünü yeniden tara ve vektör veritabanını güncelle
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

                    embeddings = OpenAIEmbeddings()
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

    # İlk Kurulum / GitHub Verilerini İndeksleme Butonu
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
                    embeddings = OpenAIEmbeddings()
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

    # Soru - Cevap Arayüzü
    if os.path.exists(DB_DIR):
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=DB_DIR, embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )

        st.markdown("---")
        user_query = st.text_input(
            "Mevzuatla ilgili sorunuzu yazın:",
            placeholder="Örn: Bu tebliğe göre şartlar nelerdir?",
        )

        if user_query:
            with st.spinner("Mevzuat taranıyor..."):
                response = qa_chain.invoke({"query": user_query})
                ans = response["result"]
                sources = response["source_documents"]

                st.markdown("### 🤖 Yanıt:")
                st.write(ans)

                with st.expander("📚 Kaynak Mevzuat Maddeleri"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Kaynak {i+1}:**")
                        st.text(doc.page_content[:300] + "...")
                        if "source" in doc.metadata:
                            st.caption(f"Dosya: {doc.metadata['source']}")
    else:
        st.info(
            "Sistemi başlatmak için sol menüden **'GitHub'daki Mevzuatları İlk Kez İndeksle'** butonuna basın."
        )
