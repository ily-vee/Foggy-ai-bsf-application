import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# run this script only when you add new PDF manuals or research papers to your knowledge_docs folder. It compiles
# them into foggy_vector_db
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


def embed_and_store_knowledge(folder_path, db_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len
    )

    all_chunks = []

    # 1. Extract and chunk text from all files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        raw_text = ""

        if filename.endswith(".txt"):
            print(f"Reading Text File: {filename}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

        elif filename.endswith(".pdf"):
            print(f"Reading PDF File: {filename}...")
            raw_text = extract_text_from_pdf(file_path)

        else:
            continue

        if not raw_text.strip():
            continue

        # Create chunks and tag them with the source filename for traceability
        file_chunks = text_splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"source": filename}]
        )
        all_chunks.extend(file_chunks)
        print(f"-> Prepared {len(file_chunks)} chunks.")

    # 2. Initialize ChromaDB and save vectors permanently to disk
    if all_chunks:
        print(f"\nSaving {len(all_chunks)} total chunks into local vector database at '{db_path}'...")

        Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=db_path
        )
        print("--- Database successfully built and saved! ---")
    else:
        print("No valid text found to process.")


if __name__ == "__main__":
    # Input folder with PDFs
    docs_folder = "knowledge_docs"
    # Destination folder for your saved database
    vector_db_folder = "foggy_vector_db"

    embed_and_store_knowledge(docs_folder, vector_db_folder)
