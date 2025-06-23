# ZoomCam Docker Image
# Multi-stage build for optimized production image

# ==================================================
# Stage 1: Base image with system dependencies
# ==================================================
FROM ubuntu:22.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # FFmpeg for streaming
    ffmpeg \
    # GStreamer for camera handling
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    # Video4Linux utilities
    v4l-utils \
    # Git for logging
    git \
    # Network tools
    curl \
    wget \
    # System monitoring
    htop \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ==================================================
# Stage 2: Poetry and dependencies
# ==================================================
FROM base as builder

WORKDIR /app

# Ensure Poetry creates the venv inside the project directory
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Poetry
RUN pip3 install poetry

# Copy only dependency files first for better caching
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev, no root)
RUN poetry install --only=main --no-root && rm -rf /tmp/poetry_cache

# ==================================================
# Stage 3: Production image
# ==================================================
FROM base as production

# Create zoomcam user
RUN groupadd --gid 1000 zoomcam \
    && useradd --uid 1000 --gid zoomcam --shell /bin/bash --create-home zoomcam

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/logs/git-repository \
    && mkdir -p /app/logs/screenshots \
    && mkdir -p /app/logs/performance \
    && mkdir -p /app/recordings \
    && mkdir -p /tmp/zoomcam_hls \
    && mkdir -p /var/lib/zoomcam

# Set permissions
RUN chown -R zoomcam:zoomcam /app \
    && chown -R zoomcam:zoomcam /tmp/zoomcam_hls \
    && chown -R zoomcam:zoomcam /var/lib/zoomcam

# Switch to zoomcam user
USER zoomcam

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV ZOOMCAM_CONFIG_DIR="/app/config"
ENV ZOOMCAM_DATA_DIR="/var/lib/zoomcam"
ENV ZOOMCAM_LOG_DIR="/app/logs"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/system/status || exit 1

# Expose ports
EXPOSE 8000

# Default command
CMD ["python", "-m", "zoomcam.main", "--config", "/app/config/user-config.yaml"]

# ==================================================
# Stage 4: Development image
# ==================================================
FROM production as development

# Switch back to root for dev tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    tmux \
    fish \
    tree \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for development
RUN pip3 install poetry

# Switch back to zoomcam user
USER zoomcam

# Install development dependencies
WORKDIR /app
RUN poetry install --no-root

# Set development environment
ENV ZOOMCAM_ENV=development
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Development command
CMD ["python", "-m", "zoomcam.main", "--config", "/app/config/user-config.yaml", "--debug"]
