FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir numpy torch flask transformers faiss-cpu PyPDF2 flask-cors boto3 python-dotenv

EXPOSE 5000

ENV NAME=TextGenerationApp

# Run app.py when the container launches
CMD ["python", "app.py"]