FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml .
COPY . .

RUN pip install uv
RUN python -m uv pip install .

EXPOSE 9696

CMD ["python", "-m", "uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]
