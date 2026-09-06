import hashlib
import re
from pathlib import Path
from typing import Iterable

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import Settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx", ".csv"}


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    return cleaned or "uploaded_document"


def calculate_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


async def save_upload(file: UploadFile, settings: Settings) -> Path:
    settings.ensure_dirs()
    target = settings.upload_dir / safe_filename(file.filename or "upload")
    target.write_bytes(await file.read())
    return target


def loader_for(file_path: Path):
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return PyPDFLoader(str(file_path))
    if extension == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")
    if extension in {".doc", ".docx"}:
        return Docx2txtLoader(str(file_path))
    if extension == ".csv":
        return CSVLoader(str(file_path))
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")


def load_and_split(file_path: Path, original_name: str, settings: Settings) -> tuple[str, list[Document]]:
    file_hash = calculate_file_hash(file_path)
    documents = loader_for(file_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text found in the document.")

    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "hash": file_hash,
                "source": original_name,
                "chunk_index": index,
            }
        )
    return file_hash, chunks


def metadata_contains_hash(metadatas: Iterable[dict], file_hash: str) -> bool:
    return any((metadata or {}).get("hash") == file_hash for metadata in metadatas)
