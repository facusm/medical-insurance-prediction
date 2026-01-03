FROM python:3.11-slim

# LightGBM suele requerir OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

RUN chmod +x docker/entrypoint.sh

CMD ["bash", "docker/entrypoint.sh"]
