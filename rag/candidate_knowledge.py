import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# =========================================================
# CONFIGURATION
# =========================================================
load_dotenv()
client = OpenAI()

RESUME_DIR = Path("data/resume")
RESUME_DIR.mkdir(parents=True, exist_ok=True)

MASTER_RESUME_PATH = RESUME_DIR / "master_resume.pdf"

CHROMA_PATH = "data/chroma"


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

resume_collection = chroma_client.get_or_create_collection(
    name="candidate_resume"
)


# =========================================================
# RESUME STORAGE
# =========================================================

def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def save_master_resume(uploaded_file):
    with open(MASTER_RESUME_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())


def resume_exists():
    return MASTER_RESUME_PATH.exists()


def get_resume_hash():
    if not resume_exists():
        return None

    with open(MASTER_RESUME_PATH, "rb") as f:
        file_bytes = f.read()

    return hashlib.sha256(
        file_bytes
    ).hexdigest()[:12]


def load_master_resume():
    if not resume_exists():
        return ""

    with open(MASTER_RESUME_PATH, "rb") as f:
        return extract_pdf_text(f)


# =========================================================
# CHUNKING
# =========================================================

def chunk_text(
    text,
    chunk_size=180,
    overlap=30
):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# INDEX RESUME
# =========================================================

def index_resume(
    resume_text,
    resume_version
):
    chunks = chunk_text(resume_text)

    existing = resume_collection.get()

    existing_metadata = existing.get(
        "metadatas",
        []
    )

    existing_versions = set()

    for metadata in existing_metadata:
        if metadata:
            version = metadata.get(
                "resume_version"
            )

            if version:
                existing_versions.add(
                    version
                )

    # Already indexed
    if resume_version in existing_versions:
        return len(chunks)

    # New resume version:
    # remove old embeddings
    if existing.get("ids"):
        resume_collection.delete(
            ids=existing["ids"]
        )

    for i, chunk in enumerate(chunks):

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        embedding = (
            response.data[0].embedding
        )

        resume_collection.add(
            ids=[
                f"{resume_version}_{i}"
            ],
            documents=[
                chunk
            ],
            embeddings=[
                embedding
            ],
            metadatas=[
                {
                    "resume_version":
                        resume_version,

                    "chunk_number":
                        i
                }
            ]
        )

    return len(chunks)


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_resume_evidence(
    job_description,
    top_k=6
):
    total_chunks = resume_collection.count()

    if total_chunks == 0:
        return []

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=job_description
    )

    query_embedding = (
        response.data[0].embedding
    )

    actual_top_k = min(
        top_k,
        total_chunks
    )

    results = resume_collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=actual_top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )

    if not documents:
        return []

    return documents[0]