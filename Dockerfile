# Use a lightweight Python image
FROM python:3.9-slim

# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy app and model files
COPY app.py /app/app.py
COPY model.joblib /app/model.joblib
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Gradio app port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]

