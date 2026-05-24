#!/bin/sh
# startup.sh — container entrypoint
#
# Vector-store resolution order:
#   1. If GCS_VECTOR_STORE_URI is set, download (overwrites baked-in copy).
#   2. Otherwise use the vector_store/ baked into the image at build time.
#
# Setting GCS_VECTOR_STORE_URI lets you refresh the knowledge base without
# rebuilding the Docker image.
#
#   gcloud run services update rag-api \
#     --update-env-vars GCS_VECTOR_STORE_URI=gs://my-rag-bucket/vector_store/

set -e

if [ -n "$GCS_VECTOR_STORE_URI" ]; then
  echo "[startup] Downloading vector store from ${GCS_VECTOR_STORE_URI} ..."
  mkdir -p vector_store
  # Use the google-cloud-storage Python client (gsutil is not in the slim image).
  python - <<PY
import os, sys
from urllib.parse import urlparse
from google.cloud import storage

uri = os.environ["GCS_VECTOR_STORE_URI"].rstrip("/")
p = urlparse(uri)
if p.scheme != "gs" or not p.netloc:
    sys.exit(f"Invalid GCS_VECTOR_STORE_URI: {uri}")
bucket_name, prefix = p.netloc, p.path.lstrip("/")
client = storage.Client()
count = 0
for blob in client.list_blobs(bucket_name, prefix=prefix):
    if blob.name.endswith("/"):
        continue
    rel = blob.name[len(prefix):].lstrip("/")
    if not rel:
        continue
    dest = os.path.join("vector_store", rel)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    blob.download_to_filename(dest)
    count += 1
print(f"[startup] Downloaded {count} object(s) from {uri}")
PY
else
  echo "[startup] GCS_VECTOR_STORE_URI not set — using vector_store/ baked into image."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
