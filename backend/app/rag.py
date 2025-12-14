from typing import Any, Dict, List, Optional

from .config import settings

# Make RAG optional so the backend can boot even if deps aren't installed yet.
try:
    from langchain_community.vectorstores import Chroma  # type: ignore
    from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    from langchain_core.documents import Document  # type: ignore

    _RAG_DEPS_OK = True
except Exception:
    Chroma = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    Document = None  # type: ignore
    _RAG_DEPS_OK = False


class RAGService:
    """Simple RAG wrapper using Chroma by default.

    You can swap out the implementation for Pinecone/Weaviate by
    changing `init_vector_store`.
    """

    def __init__(self):
        if not _RAG_DEPS_OK:
            self.embeddings = None
            self.text_splitter = None
            self.vectorstore = None
            return

        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)  # type: ignore[misc]
        self.text_splitter = RecursiveCharacterTextSplitter(  # type: ignore[misc]
            chunk_size=1500, chunk_overlap=200
        )
        self.vectorstore = self._init_vector_store()

    def _init_vector_store(self):
        # Default: local Chroma in `backend/data/chroma_db`
        if settings.VECTOR_DB_PROVIDER == "chroma":
            if not _RAG_DEPS_OK:
                return None
            return Chroma(  # type: ignore[misc]
                embedding_function=self.embeddings,
                persist_directory=settings.VECTOR_DB_DIR,
            )

        # Stubs for Pinecone/Weaviate – extend as needed
        raise NotImplementedError(
            f"VECTOR_DB_PROVIDER={settings.VECTOR_DB_PROVIDER} is not implemented yet."
        )

    def ingest_text(self, text: str, metadata: dict | None = None) -> int:
        """Split and add text into the vector store."""
        if not _RAG_DEPS_OK or not self.vectorstore or not self.text_splitter:
            return 0
        docs = self.text_splitter.split_text(text)
        documents: List[Document] = [
            Document(page_content=chunk, metadata=metadata or {}) for chunk in docs
        ]
        self.vectorstore.add_documents(documents)
        # Persist if supported (Chroma)
        try:
            self.vectorstore.persist()
        except Exception:
            pass
        return len(documents)

    def retrieve(self, query: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve top-k relevant chunks.

        `where` is a metadata filter (Chroma-style). Example:
        {"story_type": "Narrative", "channel": "Stocks", "example_id": "ex_001"}
        """
        if not _RAG_DEPS_OK or not self.vectorstore:
            return []
        search_kwargs: Dict[str, Any] = {"k": k}
        if where:
            search_kwargs["filter"] = where

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Newer LangChain retrievers are Runnables and use .invoke();
        # older versions expose .get_relevant_documents. Support both.
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)  # type: ignore[return-value]

        return retriever.invoke(query)  # type: ignore[return-value]


rag_service = RAGService()


