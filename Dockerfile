# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies including git and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the checkpoints directory
RUN mkdir -p checkpoints

# Make the download script executable
RUN chmod +x download_ckpts.sh

# Run the checkpoint download script
RUN ./download_ckpts.sh

# Move the downloaded checkpoints to the checkpoints directory
RUN mv *.pt checkpoints/

# Set the entry point to your Python script
CMD ["python", "sam2_serverless.py"]