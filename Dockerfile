# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Rust
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set writable Cargo and Rustup directories
ENV CARGO_HOME=/app/.cargo
ENV RUSTUP_HOME=/app/.rustup

COPY . .

RUN chmod +x build.sh
RUN ./build.sh

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]