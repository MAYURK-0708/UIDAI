FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=api_server.py

# Run the startup script
CMD ["python", "start_all_services.py"]
