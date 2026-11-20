FROM python:3.11-slim
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-semantic.txt
COPY app ./app
COPY main.py ./
RUN useradd --create-home rag && mkdir -p /app/data /var/data && chmod 777 /app/data /var/data && chown -R rag:rag /app /var/data
USER rag
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
