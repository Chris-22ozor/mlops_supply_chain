FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY api ./api
COPY core ./core
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY reports ./reports

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[api]

RUN python scripts/train.py \
    && python scripts/deploy_staging.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
