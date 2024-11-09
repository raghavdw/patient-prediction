# Use a lightweight Python image
FROM python:3.9-slim

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the port Gradio will run on
EXPOSE 7860

# Start the Gradio app
CMD ["python", "app.py"]
