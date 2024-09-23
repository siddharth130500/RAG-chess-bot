# Chess Bot Application

This is a Chess Bot application that uses a combination of natural language processing models, including DistilBERT for embeddings and LLaMA 3 for text generation, to answer chess-related questions. The bot retrieves relevant information from a PDF textbook on chess, chunks the text, retrieves the most relevant pieces using FAISS, and generates a response using the LLaMA 3 model through AWS Bedrock.

## Features

- **Text Extraction**: Extracts text from a chess textbook in PDF format.
- **Text Chunking**: Splits the extracted text into manageable chunks for processing.
- **FAISS Indexing**: Retrieves top-k relevant document chunks based on query embeddings.
- **LLaMA 3 Integration**: Uses LLaMA 3 to generate responses to chess-related queries.
- **Flask Frontend**: A simple web-based frontend where users can input chess queries and receive responses.

## How It Works

1. Extracts text from a chess PDF textbook using PyPDF2.
2. Chunks the text into smaller segments to work within model constraints.
3. Generates embeddings for the text chunks using DistilBERT.
4. Stores embeddings in a FAISS index to enable fast similarity search.
5. Retrieves the top-k similar chunks based on the user’s query.
6. Calls the LLaMA 3 model via AWS Bedrock to generate a detailed response based on the query and retrieved context.

## Technologies Used

- **Flask**: Web framework to serve the app.
- **DistilBERT**: For generating embeddings.
- **LLaMA 3**: For generating detailed text responses via AWS Bedrock.
- **FAISS**: For efficient similarity search and retrieval.
- **AWS Bedrock**: For hosting and invoking the LLaMA 3 model.
- **Docker**: Containerization for easy deployment.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/chess-bot-app.git
cd chess-bot-app
