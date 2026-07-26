FROM python:3.11-slim
WORKDIR /app

# onnxruntime + numpy replace torch for inference. The gateway now runs the
# simulation in-process (live/runner.py), and a torch install would add ~800 MB
# to the image and ~300 MB resident — more than a 512 MB free instance can
# spare next to uvicorn. See agents/policy_runtime.py.
RUN pip install --no-cache-dir \
      fastapi uvicorn websockets redis kafka-python httpx \
      numpy onnxruntime

COPY comms/ ./comms/
COPY visuals/ ./visuals/
# Needed by the in-process simulation: the agent/funnel/social-graph code, the
# persona split, the exported policy and the product vocabulary.
COPY agents/ ./agents/
COPY data/ ./data/
COPY live/ ./live/
COPY policy_rl.onnx vocab.json ./
COPY backend/gateway/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
