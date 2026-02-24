# Multi-stage build for the stat-arb pairs trading system.
# Target: mrm-x86 (Ubuntu VM on Proxmox, Tailscale IP 100.87.161.31)

# --- Stage 1: Builder (install deps into a venv) ---
FROM python:3.11-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml /build/pyproject.toml
COPY src/ /build/src/

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .


# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

# Create non-root user
RUN groupadd -r statarb && useradd -r -g statarb -m statarb

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source and config
WORKDIR /app
COPY src/ src/
COPY config/default.yaml config/default.yaml
COPY scripts/ scripts/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Data directory for SQLite DB and tokens
RUN mkdir -p /data && chown statarb:statarb /data
VOLUME /data

# Default env: paper mode, SQLite in /data
ENV STAT_ARB_CONFIG=/app/config/default.yaml \
    STAT_ARB_DB_URL="sqlite:////data/stat_arb.db" \
    SCHWAB_TOKENS_DB="/data/tokens.db" \
    BROKER_MODE=paper \
    PYTHONPATH=/app/src

USER statarb

ENTRYPOINT ["/entrypoint.sh"]
CMD ["run-live"]
