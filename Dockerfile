# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:2.10.1-gpu

# Install ffmpeg for video processing with OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Install USBMD module
RUN python -m pip install -e .
