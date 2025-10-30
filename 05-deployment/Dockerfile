# Start from the prepared image that already contains pipeline_v2.bin
FROM agrigorev/zoomcamp-model:2025

# bring uv in for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code
ENV PATH="/code/.venv/bin:$PATH"

# install deps from pyproject (use lock if present)
COPY pyproject.toml ./
# COPY uv.lock ./   # uncomment if you have it
RUN uv sync --frozen || uv sync

# copy the FastAPI app; use the pipeline inside the base image
COPY predict.py ./
ENV MODEL_PATH=/code/pipeline_v2.bin

EXPOSE 9696
ENTRYPOINT ["uvicorn","predict:app","--host","0.0.0.0","--port","9696"]