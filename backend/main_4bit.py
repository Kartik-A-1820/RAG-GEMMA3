import os
import re
import hashlib
import traceback
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

# Disable incompatible attention mode
os.environ["PYTORCH_SDP_ATTENTION"] = "0"
load_dotenv()

# Load model & tokenizer once globally
model_id = "google/gemma-3-1b-it"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
gemma_tokenizer = AutoTokenizer.from_pretrained(model_id)
gemma_model = Gemma3ForCausalLM.from_pretrained(model_id, quantization_config=quant_config).eval()

# Initialize vectorstore
vectorstore = Chroma(
    collection_name="document_vector_collection",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory="./chroma_langchain_db"
)

if not os.path.exists("temp"):
    os.makedirs("temp")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def extract_answer(conversation_text: str) -> str:
    match = re.search(r"<start_of_turn>model\s*(.*)", conversation_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        answer = re.split(r"<end_of_turn>", answer)[0].strip()
        return answer
    return "Answer not found."

def calculate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    ext = file.filename.lower()
    if ext.endswith(".pdf"):
        loader = PyPDFLoader(file_location)
    elif ext.endswith(".txt"):
        loader = TextLoader(file_location)
    elif ext.endswith((".doc", ".docx")):
        loader = Docx2txtLoader(file_location)
    elif ext.endswith(".csv"):
        loader = CSVLoader(file_location)
    else:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_hash = calculate_file_hash(file_location)
    existing_hashes = [item.get("hash") for item in vectorstore.get().get("metadatas", [])]
    if file_hash in existing_hashes:
        os.remove(file_location)
        return {"message": "Document already ingested.", "status": "already_ingested"}

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    split_docs = text_splitter.split_documents(docs)

    if not split_docs:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="No valid text found in the document.")

    uuids = [str(uuid4()) for _ in split_docs]
    metadatas = [doc.metadata for doc in split_docs]
    for metadata in metadatas:
        metadata["hash"] = file_hash
        metadata["source"] = file.filename

    texts = [doc.page_content for doc in split_docs]
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=uuids)
    os.remove(file_location)

    return {"message": "File ingested successfully.", "num_documents": len(split_docs)}

@app.post("/query/local")
async def query_local(data: QueryRequest):
    try:
        query = data.query

        # Perform similarity search on vectorstore
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return {
                "answer": "Not found relevant answer.",
                "references": []
            }

        # Construct reference text for the system message
        reference_text = "\n\n".join([
            f"Source: {doc.metadata.get('source', f'Document {i+1}')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
            for i, doc in enumerate(results)
        ])

        # System message to enforce grounded answer only
        system_instruction = (
            "You are a helpful assistant. Use ONLY the provided references below to answer the user's question. "
            "If the answer is not present or cannot be determined from the references, reply with: "
            "'Not found in the provided reference.'\n\n"
            "References:\n" + reference_text
        )

        # Prepare prompt for Gemma model
        prompt = [[
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [{"type": "text", "text": query}]},
        ]]

        # Tokenize input
        inputs = gemma_tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(gemma_model.device)

        # Generate output
        with torch.inference_mode():
            outputs = gemma_model.generate(**inputs, max_new_tokens=1024)

        decoded_output = gemma_tokenizer.batch_decode(outputs)[0]
        answer_text = extract_answer(decoded_output).strip()

        # Build reference list with page number
        references = [{
            "source": doc.metadata.get("source", f"Doc {i+1}"),
            "page": doc.metadata.get("page", "N/A"),
            "content": doc.page_content
        } for i, doc in enumerate(results)]

        return {
            "answer": answer_text,
            "references": references
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred during processing.")


@app.post("/summarize_pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    ext = file.filename.lower()
    if ext.endswith(".pdf"):
        loader = PyPDFLoader(file_location)
    elif ext.endswith(".txt"):
        loader = TextLoader(file_location)
    elif ext.endswith((".doc", ".docx")):
        loader = Docx2txtLoader(file_location)
    elif ext.endswith(".csv"):
        loader = CSVLoader(file_location)
    else:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    documents = loader.load()
    if not documents:
        os.remove(file_location)
        return {"summary": "Failed to read document content."}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    chunk_texts = [chunk.page_content for chunk in chunks]

    all_summaries = []
    for i in range(0, len(chunk_texts), 3):  # Batch 3 chunks at once
        batch_text = "\n\n".join(chunk_texts[i:i+3])
        prompt = [[
            {"role": "system", "content": [{"type": "text", "text": "You are a text summarization engine. Provide ONLY a concise summary of the following text. Do not include greetings, explanations, or questions."}]},
            {"role": "user", "content": [{"type": "text", "text": batch_text}]},
        ]]

        inputs = gemma_tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(gemma_model.device)

        try:
            with torch.inference_mode():
                output = gemma_model.generate(**inputs, max_new_tokens=2048)
            decoded = gemma_tokenizer.batch_decode(output)[0]
            summary = extract_answer(decoded).strip()
            all_summaries.append(summary)
        except:
            continue

    merged_summary_prompt = [[
        {"role": "system", "content": [{"type": "text", "text": "You are a text summarization engine. Combine the following summaries into a single, concise final summary of the original document. Provide ONLY the final summary text. Do not include greetings, explanations, or questions."}]},
        {"role": "user", "content": [{"type": "text", "text": "\n\n".join(all_summaries)}]}
    ]]

    inputs = gemma_tokenizer.apply_chat_template(
        merged_summary_prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(gemma_model.device)

    try:
        with torch.inference_mode():
            final_output = gemma_model.generate(**inputs, max_new_tokens=4096)
        final_summary = extract_answer(gemma_tokenizer.batch_decode(final_output)[0]).strip()
    except:
        final_summary = "\n".join(all_summaries)

    os.remove(file_location)
    return {"summary": final_summary}
