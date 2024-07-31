# Use a base image with Python and Git
FROM python:3.9-alpine

# Install required packages
RUN apk add --no-cache git

# Set working directory
WORKDIR /app

# Copy the Flask app to the container
COPY host_docs.py /app/host_docs.py

# Install Flask
RUN pip install flask Authlib requests gunicorn

# Expose port 6000
EXPOSE 6001

# Default entry point to python script
ENTRYPOINT ["gunicorn", "-w", "2", "-b", "0.0.0.0:6001", "host_docs:app"]
