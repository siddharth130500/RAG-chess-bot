# Use a Python slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files for building the application
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define environment variables (excluding credentials)
ENV NAME=TextGenerationApp

# Run app.py when the container launches
CMD ["python", "app.py"]
