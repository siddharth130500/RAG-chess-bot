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
git clone https://github.com/siddharth130500/RAG_chess_bot.git
cd chess-bot
```

### 2. Install Dependencies

Ensure you have Python 3.10 or later installed. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

The application requires AWS credentials to interact with the LLaMA 3 model via AWS Bedrock. Create a .env file in the root directory of the project with the following:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

### 4. Run the Application

Start the Flask server locally:
```bash
python app.py
```
The application will be accessible at http://localhost:5000.

### 5. Docker Setup

The Docker image for the Chess Bot application is available on DockerHub.
# DockerHub Image
You can pull the pre-built Docker image from DockerHub:
```bash
docker pull siddharth130500/chess-bot-app:latest
```

# Run the Docker Container
Ensure that your .env file is mapped into the Docker container for AWS authentication:
```bash
docker run -d -p 5000:5000 -v /path/to/your/.env:/app/.env siddharth130500/chess-bot-app
```

### 6. Usage

# 1. Open a web browser and go to http://localhost:5000.
# 2. Enter a chess-related question in the input field.
# 3. Submit the query and wait for the response generated by the LLaMA 3 model.


