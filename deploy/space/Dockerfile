# lobot public demo: replay mode, read-only, no weights, no API key.
# build.sh puts the code in src/ and the recorded jobs in jobs/.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.12.13

RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user src/ /app/
COPY --chown=user jobs/ /app/jobs/

# CPU torch: replay mode never loads the model, but the server imports torch.
RUN uv venv --python 3.12 /app/.venv \
 && VIRTUAL_ENV=/app/.venv uv pip install --torch-backend cpu --no-cache -r pyproject.toml \
 && VIRTUAL_ENV=/app/.venv uv pip install --no-deps --no-cache -e . \
 && chown -R user /app

USER user
ENV MODEL_LOBOTOMY_REPLAY=1 \
    LOBOT_PUBLIC_DEMO=1 \
    LOBOT_MODEL=meta-llama/Llama-3.2-3B-Instruct \
    LOBOT_HOST=0.0.0.0 \
    PORT=7860 \
    LOBOT_HOME=/tmp/lobot \
    HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["/app/.venv/bin/python", "-m", "model_lobotomy.server"]
