import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from flask import Flask, request, jsonify, render_template
import faiss
import PyPDF2
import os
import boto3
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Load embedding model and tokenizer
embedding_model_name = "distilbert-base-uncased"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

# Set the model ID for Llama 3
model_id = "meta.llama3-8b-instruct-v1:0"

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to chunk text into smaller segments
def chunk_text(text, max_length=512, model='distilbert'):
    words = text.split()  # Split the text into words
    current_chunk = []
    current_length = 0
    chunks = []

    for word in words:
        word_length = len(embedding_tokenizer.tokenize(word))  # Estimate the token length of the word
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to get embeddings
def get_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

# Path to save/load FAISS index
faiss_index_file = "faiss_index.idx"
faiss_embeddings_file = "chunk_embeddings.npy"

# Function to save FAISS index
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# Function to load FAISS index
def load_faiss_index(file_path):
    return faiss.read_index(file_path)

# Extract text from PDF
pdf_text = extract_text_from_pdf("./Chess.pdf")
# Ensure chunks are within the model's limit for DistilBERT
distilbert_chunk_size = 512  # Max token length for DistilBERT
document_chunks = chunk_text(pdf_text, max_length=distilbert_chunk_size)

# Check if FAISS index and embeddings are already saved
if os.path.exists(faiss_index_file) and os.path.exists(faiss_embeddings_file):
    print("Loading FAISS index from file...")
    # Load the embeddings and FAISS index from disk
    chunk_embeddings = np.load(faiss_embeddings_file)
    idx = load_faiss_index(faiss_index_file)
else:
    print("FAISS index not found, creating a new one...")
    
    # Generate embeddings for each chunk
    chunk_embeddings = np.array([get_embedding(chunk) for chunk in document_chunks])

    # Build the FAISS index
    idx = faiss.IndexFlatL2(chunk_embeddings.shape[1])  # L2 distance metric
    idx.add(chunk_embeddings)

    # Save the FAISS index and embeddings to disk
    save_faiss_index(idx, faiss_index_file)
    np.save(faiss_embeddings_file, chunk_embeddings)
    print("FAISS index created and saved.")

# Function to retrieve top-k closest document chunks
def retrieve(query_embedding, top_k=5):
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = idx.search(query_embedding, top_k)
    return [document_chunks[i] for i in indices[0]]

# Function to call Llama 3 through Boto3
def call_llama3_boto3(prompt):
    formatted_prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0.7,
    }

    request = json.dumps(native_request)

    try:
        streaming_response = client.invoke_model_with_response_stream(
            modelId=model_id, body=request
        )
        response_text = ""
        for event in streaming_response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if "generation" in chunk:
                response_text += chunk["generation"]

        return response_text.strip()

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return ""

# Serve frontend
@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/retrieve', methods=['POST'])
def retrieve_and_generate():
    data = request.json
    prompt = data.get("query", "")
    print(f"Received query: {prompt}")

    query_embedding = get_embedding(prompt)
    retrieved_chunks = retrieve(query_embedding, 5)

    # Combine retrieved chunks into a single context
    retrieved_context = " ".join(retrieved_chunks)
    enhanced_prompt = "Answer the query in detail. <|query_start|>" + prompt + "<|query_end|>" + " <|context_start|>" + retrieved_context + "<|context_end|>"
    
    # Call Llama 3 model through Boto3
    final_output = call_llama3_boto3(enhanced_prompt)
    return jsonify({"generated_text": final_output})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
