FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DOCKER_SANDBOX_WORKDIR=/workspace \
    HOME=/home/agent \
    PATH=/home/agent/.local/bin:${PATH}

# Basic tools commonly needed by a coding agent. The agent workspace is
# mounted separately at /workspace when the session container is started.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        build-essential \
        cmake \
        gfortran \
        git \
        libffi-dev \
        libjpeg-dev \
        libopenblas-dev \
        liblapack-dev \
        liblzma-dev \
        libncurses-dev \
        libpng-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        libyaml-dev \
        ninja-build \
        passwd \
        pkg-config \
        python3-dev \
        tk-dev \
        util-linux \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the Python packaging toolchain in the image. SWE-bench projects
# frequently build native extensions and some tasks use --no-build-isolation,
# so these tools must already be available in the read-only runtime image.
RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

ARG AGENT_UID=1000
ARG AGENT_GID=1000

RUN groupadd --gid "${AGENT_GID}" agent \
    && useradd \
        --uid "${AGENT_UID}" \
        --gid "${AGENT_GID}" \
        --create-home \
        --shell /bin/bash \
        agent \
    && mkdir -p /opt/sandbox-tools /workspace /external \
    && chown -R agent:agent /opt/sandbox-tools /workspace /home/agent /external

COPY docker/sandbox-tools/file_ops.py /opt/sandbox-tools/file_ops.py

RUN chown agent:agent /opt/sandbox-tools/file_ops.py

WORKDIR ${DOCKER_SANDBOX_WORKDIR}

# Defense in depth: runtime also sets user="agent" when creating the
# session container.
USER agent

# The session manager starts the container and keeps it alive. Tool calls
# execute commands inside this same container with `docker exec`.
CMD ["sleep", "infinity"]
