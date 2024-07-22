FROM tensorflow/tensorflow:2.7.0-gpu

ENV GOOGLE_APPLICATION_CREDENTIALS "/app/credentials.json"

WORKDIR /app

COPY ../../requirements.in .
RUN pip install -U --no-cache-dir -r requirements.in

COPY ../.. .
