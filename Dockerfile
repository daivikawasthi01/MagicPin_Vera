FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and seed dataset
COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn bot:app --host 0.0.0.0 --port ${PORT:-8080}"]
