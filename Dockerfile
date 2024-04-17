ARG PYTHON_VERSION="3.11"

###############################################################################
# 1. Install dependencies
###############################################################################
# poetry setup code based on https://github.com/thehale/docker-python-poetry (does not provide multiarch images)
FROM python:${PYTHON_VERSION} AS build-python
ARG POETRY_VERSION="1.6.1"

ENV POETRY_VERSION=${POETRY_VERSION}
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends curl
# Install Poetry via the official installer: https://python-poetry.org/docs/master/#installing-with-the-official-installer
# This script respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -
# only install dependencies into project virtualenv
WORKDIR /app
COPY requirements.linux-cpu.txt pyproject.toml poetry.lock ./
RUN poetry run python -m pip install -r requirements.linux-cpu.txt
RUN poetry install --only main --no-root --no-cache

###############################################################################
# 2. Final, minimal image that starts the inference server daemon
###############################################################################
FROM python:${PYTHON_VERSION}-slim
ARG PYTHON_VERSION

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY ./bpm_ai_inference/ ./bpm_ai_inference/
COPY --from=build-python /app/.venv/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
RUN apt-get update  \
    && apt-get install -y --no-install-recommends curl tesseract-ocr poppler-utils  \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY init.py .
CMD ["python3", "init.py", "python -m bpm_ai_inference.daemon"]