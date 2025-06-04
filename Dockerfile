FROM python:3.10-slim

# Install system packages
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app
WORKDIR /app

# Create needed directories
RUN mkdir -p /data /output /server_storage

# Entry point
CMD ["python", "run.py"]
