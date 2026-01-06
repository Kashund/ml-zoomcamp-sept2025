FROM python:3.11-slim

# Keeps logs unbuffered and avoids .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Default to the winning baseline model
ENV MODEL_BACKEND=rf

WORKDIR /app

# Install deps first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . /app

# Default API port (matches README / Makefile)
EXPOSE 9696

# Start FastAPI via uvicorn.
# Note: we reference src.serve:app because the capstone layout uses src/serve.py
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "9696"]
