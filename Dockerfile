##############################################################
# House Price Prediction – FastAPI Backend
# Build:   docker build -t house-price-api .
# Run:     docker run -p 8000:8000 house-price-api
##############################################################
FROM python:3.11-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY model/   ./model/
COPY data/    ./data/

# Expose port (Render injects $PORT; default 8000 for local)
ENV PORT=8000
EXPOSE $PORT

# Switch to non-root user
USER appuser

# Start the FastAPI application
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
