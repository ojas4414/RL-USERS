FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn websockets redis kafka-python httpx
COPY comms/ ./comms/
COPY visuals/ ./visuals/
COPY backend/gateway/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
