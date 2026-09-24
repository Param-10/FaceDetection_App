# syntax=docker/dockerfile:1

FROM node:24-bookworm-slim AS frontend-build
WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY index.html vite.config.js tailwind.config.js postcss.config.cjs ./
COPY src ./src
RUN npm run build


FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=10000

# OpenCV and TensorFlow need these runtime libraries on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY app.py face_detection_model.py ./
COPY --from=frontend-build /build/dist ./dist

EXPOSE 10000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 1 --timeout 600 app:app"]
