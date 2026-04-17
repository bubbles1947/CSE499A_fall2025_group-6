"""
RAG service — PDF ingestion, embedding, and retrieval via ChromaDB + LangChain.
Embeddings are generated locally using ChromaDB's built-in SentenceTransformer
(all-MiniLM-L6-v2) — no separate embeddings server needed.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

settings = get_settings()

# Lazy singleton — loads model once, reuses across requests
_embedding_fn: SentenceTransformerEmbeddingFunction | None = None


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embedding_fn


_chroma_client = None


def _get_chroma_client():
    """Return a persistent embedded ChromaDB client (no external server needed)."""
    global _chroma_client
    if _chroma_client is None:
        import os
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_data")
        os.makedirs(data_path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=data_path)
    return _chroma_client


async def process_pdf(file_path: str, collection_name: str) -> None:
    """Load a PDF, split into chunks, embed, and store in ChromaDB."""
    # Load PDF pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("PDF produced no text chunks — it may be empty or image-only")

    # Extract text and metadata
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"page": chunk.metadata.get("page", 0), "source": file_path} for chunk in chunks]
    ids = [f"{collection_name}_chunk_{i}" for i in range(len(texts))]

    # Store in ChromaDB — embedding_fn generates embeddings automatically
    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_fn(),
    )
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )


async def query_document(
    collection_name: str,
    query: str,
    model_name: str,
    n_results: int = 5,
) -> tuple[str, list[str]]:
    """
    Retrieve relevant chunks from ChromaDB and ask the LLM to answer.
    Returns (answer, source_chunks).
    """
    import httpx

    # Retrieve from ChromaDB — embedding_fn handles query embedding automatically
    client = _get_chroma_client()
    collection = client.get_collection(
        name=collection_name,
        embedding_function=_get_embedding_fn(),
    )
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    source_texts = results["documents"][0] if results["documents"] else []

    if not source_texts:
        return "No relevant information found in the document.", []

    # Build prompt with context
    context = "\n\n---\n\n".join(source_texts)
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided document context. "
        "Only use information from the context below. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}"
    )

    # Resolve model server URL
    from app.routers.chat import MODEL_SERVERS
    server_info = MODEL_SERVERS.get(model_name)
    if not server_info:
        raise ValueError(f"Unknown model: {model_name}")

    # Call LLM (non-streaming for RAG)
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http_client:
        resp = await http_client.post(
            f"{server_info['url']}/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 2048,
                "temperature": 0.3,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]

    return answer, source_texts
