# This Dockerfile builds a container running fibermorph
FROM python:3.11-slim
RUN pip install poetry
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi
COPY . /app
CMD ["poetry", "run", "fibermorph"]
