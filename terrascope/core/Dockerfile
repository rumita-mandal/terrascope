   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
       && pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 7860
   CMD ["python", "-m", "uvicorn", "terrascope.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
