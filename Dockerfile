FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
