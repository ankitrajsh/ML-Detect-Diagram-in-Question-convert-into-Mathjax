# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Install system dependencies for building (including libs Pillow/OpenCV need)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    tcl-dev \
    tk-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and tools
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install CPU PyTorch wheels explicitly (uses PyTorch's CPU index)
# Pin a compatible CPU wheel to satisfy requirements; adjust versions if needed
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 || true

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libjpeg8 \
    zlib1g \
    libfreetype6 \
    liblcms2-2 \
    libwebp6 \
    libopenjp2-7 \
    libtcl8.6 \
    libtk8.6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for API (FastAPI default)
EXPOSE 8000

# Health check (optional; requires requests to be installed in requirements)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)"

# Use the startup script
CMD ["python", "start.py"]