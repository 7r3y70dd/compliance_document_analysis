# Dockerfile
# GPU-ready base with PyTorch + CUDA 12.1 (adjust tag as needed)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Optional: system deps for building any remaining wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency list & setup script first (better Docker layer caching)
COPY requirements.txt ./requirements.txt
COPY docker/setup.sh ./docker/setup.sh

# Run deterministic setup
RUN chmod +x ./docker/setup.sh && ./docker/setup.sh

# Now copy the rest of the app
COPY . /app

EXPOSE 8000

# Entrypoint: start FastAPI via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
