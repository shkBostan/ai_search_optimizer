# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "src.search_api:app", "-b", "0.0.0.0:5000", "--workers", "1"]