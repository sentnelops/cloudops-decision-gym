# Use a slim Python image for a smaller footprint
FROM python:3.11.9-slim-bookworm

# Set environment variables for better Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# First copy only requirements to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser

# Copy the rest of the application
# We do this after dependency installation for faster rebuilds
COPY --chown=appuser:appuser . .

# Expose the Gradio / OpenEnv port
EXPOSE 7860

# Run the app using the root wrapper which handles path setup
CMD ["python", "main.py"]
