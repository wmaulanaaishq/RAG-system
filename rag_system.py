import os
import chromadb
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import requests

# Konstanta Konfigurasi API
AIML_API_KEY = "c2837504c54f21e006ccd36e45a29656"
AIML_BASE_URL = "https://api.aimlapi.com/v1"

# Konstanta Konfigurasi ChromaDB Cloud
CHROMA_API_KEY = "ck-5yjCg8hNWtu26fJw5FUSu2vdTRY1wYVK1HRdnFVWyWmj"
CHROMA_TENANT = "78e6fff5-31a9-44c9-a813-8ed1ed673b73"
CHROMA_DATABASE = "OOP_RAG"

class AIMLEmbeddings(Embeddings):
    """Custom Embeddings wrapper untuk AIML API agar menghindari error validasi input."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{AIML_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {AIML_API_KEY}", "Content-Type": "application/json"},
            json={"model": "text-embedding-3-small", "input": texts}
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        return [item["embedding"] for item in data]
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class DocumentManager:
    """Mengelola pemuatan dan pemotongan (chunking) dokumen teks."""
    
    def __init__(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self._file_path = file_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load_and_split(self) -> List[Document]:
        """Memuat dokumen dan membaginya menjadi potongan teks (chunks)."""
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"File '{self._file_path}' tidak ditemukan.")
            
        loader = TextLoader(self._file_path, encoding="utf-8")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap
        )
        return text_splitter.split_documents(documents)


class VectorStoreManager:
    """Mengelola konversi teks ke vektor dan penyimpanannya menggunakan ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection"):
        self._collection_name = collection_name
        
        # Inisialisasi model Embeddings kustom
        self._embeddings = AIMLEmbeddings()
        
        # Menginisialisasi client Cloud ChromaDB
        self._chroma_client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
        )
        self._vector_store = None

    def store_documents(self, chunks: List[Document]) -> None:
        """Menyimpan potongan dokumen ke dalam ChromaDB vector store."""
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            client=self._chroma_client,
            collection_name=self._collection_name
        )

    def get_retriever(self, top_k: int = 3):
        """Mendapatkan objek retriever untuk proses pencarian kesamaan."""
        if self._vector_store is None:
            # Mengaitkan ke koleksi yang sudah ada jika belum diinisialisasi di instance ini
            self._vector_store = Chroma(
                client=self._chroma_client,
                collection_name=self._collection_name,
                embedding_function=self._embeddings
            )
        return self._vector_store.as_retriever(search_kwargs={"k": top_k})


class RAGSystem:
    """Orkestrator RAG yang menghubungkan Document, VectorStore, dan LLM."""
    
    def __init__(self, doc_manager: DocumentManager, vector_manager: VectorStoreManager):
        self._doc_manager = doc_manager
        self._vector_manager = vector_manager
        
        # Inisialisasi LLM menggunakan API aimlapi
        self._llm = ChatOpenAI(
            api_key=AIML_API_KEY,
            base_url=AIML_BASE_URL,
            model="gpt-4o",
            temperature=0
        )
        
        # Prompt dasar untuk RAG System
        self._prompt_template = PromptTemplate.from_template(
            "Gunakan konteks berikut untuk menjawab pertanyaan di bagian akhir. "
            "Jika kamu tidak tahu jawabannya berdasarkan konteks, katakan saja kamu tidak tahu.\n\n"
            "Konteks:\n{context}\n\n"
            "Pertanyaan: {question}\n\n"
            "Jawaban:"
        )

    def initialize_knowledge_base(self) -> None:
        """Menyiapkan basis pengetahuan dengan memuat dan memproses dokumen."""
        chunks = self._doc_manager.load_and_split()
        self._vector_manager.store_documents(chunks)
        print(f"Berhasil memproses dan menyimpan {len(chunks)} potongan dokumen ke vector database.")

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Memformat daftar dokumen menjadi satu string konteks tunggal."""
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> str:
        """Menerima pertanyaan dan menghasilkan jawaban menggunakan rantai RAG."""
        retriever = self._vector_manager.get_retriever()
        
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self._prompt_template
            | self._llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(question)


if __name__ == "__main__":
    INPUT_FILE = "materi_presentasi.txt"
    COLLECTION_NAME = "materi_oop_collection"
    
    # 1. Instansiasi komponen sistem
    document_manager = DocumentManager(file_path=INPUT_FILE)
    vector_manager = VectorStoreManager(collection_name=COLLECTION_NAME)
    
    # 2. Instansiasi orchestrator
    rag_system = RAGSystem(
        doc_manager=document_manager,
        vector_manager=vector_manager
    )
    
    # 3. Opsional: Tambahkan dokumen awal (jika belum pernah dimasukkan)
    print("Memulai setup sistem RAG...")
    try:
        rag_system.initialize_knowledge_base()
    except Exception as e:
        print(f"Melewati inisialisasi awal (file tidak ditemukan atau sudah ada): {e}")

    # 4. Proses Input Interaktif
    print("\n" + "="*50)
    print("🤖 Selamat datang di RAG Assistant!")
    print("Ketik 'keluar' atau 'exit' untuk berhenti.")
    print("Ketik 'tambah file' untuk memasukkan file baru ke database.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\nKamu: ")
            if user_input.lower() in ['keluar', 'exit', 'quit']:
                print("AI: Terima kasih! Sampai jumpa.")
                break
                
            if user_input.lower() == 'tambah file':
                file_baru = input("Masukkan nama file teks (contoh: materi_tambahan.txt): ")
                try:
                    doc_baru = DocumentManager(file_path=file_baru)
                    chunks_baru = doc_baru.load_and_split()
                    vector_manager.store_documents(chunks_baru)
                    print(f"AI: Berhasil mempelajari file '{file_baru}'!")
                except Exception as e:
                    print(f"AI: Gagal membaca file. Pastikan nama/path benar. Error: {e}")
                continue
            
            if not user_input.strip():
                continue
                
            print("AI: Berpikir...")
            answer = rag_system.ask(user_input)
            print(f"AI: {answer}")
            
        except KeyboardInterrupt:
            print("\nAI: Program dihentikan paksa. Sampai jumpa!")
            break
        except Exception as e:
            print(f"\nAI: Terjadi kesalahan: {e}")
