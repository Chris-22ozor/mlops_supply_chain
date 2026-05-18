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

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

