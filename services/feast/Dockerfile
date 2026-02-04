FROM python:3.11-slim

WORKDIR /app

# Install core requirements so the Feast CLI and repo are available.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/services/feast/feature_repo

# Default to a no-op that keeps the container alive; Feast commands can be
# executed via `docker exec` or overridden at runtime.
CMD ["sleep", "infinity"]

