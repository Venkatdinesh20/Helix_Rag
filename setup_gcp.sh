#!/bin/bash
# setup_gcp.sh — One-time GCP infrastructure setup for the RAG pipeline.
#
# Run this ONCE before your first Cloud Build / Cloud Run deployment.
# Safe to re-run — most gcloud commands are idempotent.
#
# Prerequisites:
#   1. gcloud CLI installed  →  https://cloud.google.com/sdk/docs/install
#   2. Logged in            →  gcloud auth login
#   3. Project set          →  gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   chmod +x setup_gcp.sh
#   ./setup_gcp.sh

set -euo pipefail

# ── Config — edit these if needed ────────────────────────────────────────────
REGION="us-central1"
SERVICE_NAME="rag-api"
REPOSITORY="rag-repo"
BUCKET_SUFFIX="rag-data"          # bucket will be  PROJECT_ID-rag-data
SECRET_NAME="openai-api-key"
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
  echo "ERROR: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

BUCKET_NAME="${PROJECT_ID}-${BUCKET_SUFFIX}"

echo ""
echo "=================================================="
echo " RAG Pipeline — GCP Infrastructure Setup"
echo " Project : $PROJECT_ID"
echo " Region  : $REGION"
echo "=================================================="
echo ""

# ── Step 1: Enable required APIs ─────────────────────────────────────────────
echo "[1/6] Enabling GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  --project="$PROJECT_ID"
echo "      APIs enabled."

# ── Step 2: Create Artifact Registry repository ───────────────────────────────
echo "[2/6] Creating Artifact Registry repository: $REPOSITORY ..."
gcloud artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION" \
  --description="RAG pipeline Docker images" \
  --project="$PROJECT_ID" 2>/dev/null || echo "      (already exists — skipping)"

# ── Step 3: Create GCS bucket for documents + vector store ───────────────────
echo "[3/6] Creating GCS bucket: gs://$BUCKET_NAME ..."
gcloud storage buckets create "gs://$BUCKET_NAME" \
  --location="$REGION" \
  --project="$PROJECT_ID" 2>/dev/null || echo "      (already exists — skipping)"

# Create folder structure inside the bucket
gcloud storage objects compose "gs://$BUCKET_NAME/docs/" 2>/dev/null || true
echo "      Bucket ready: gs://$BUCKET_NAME"
echo "      Upload docs  : gsutil -m cp data/raw/* gs://$BUCKET_NAME/docs/"
echo "      Upload index : gsutil -m cp vector_store/* gs://$BUCKET_NAME/vector_store/"

# ── Step 4: Create Secret Manager secret for OpenAI API key ──────────────────
echo "[4/6] Creating Secret Manager secret: $SECRET_NAME ..."
if gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" &>/dev/null; then
  echo "      (already exists — skipping)"
  echo "      To update the key: echo -n 'sk-...' | gcloud secrets versions add $SECRET_NAME --data-file=-"
else
  echo ""
  echo "      Paste your OpenAI API key and press ENTER, then Ctrl+D:"
  gcloud secrets create "$SECRET_NAME" \
    --data-file=- \
    --project="$PROJECT_ID"
  echo "      Secret created."
fi

# ── Step 5: Grant Cloud Build permission to deploy to Cloud Run ───────────────
echo "[5/6] Granting IAM roles to Cloud Build service account..."
CB_SA="${PROJECT_ID}@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/run.admin" --quiet

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/iam.serviceAccountUser" --quiet

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/secretmanager.secretAccessor" --quiet

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/storage.objectViewer" --quiet

echo "      IAM roles granted."

# ── Step 6: Connect Cloud Build trigger to your repo ─────────────────────────
echo "[6/6] Cloud Build trigger"
echo ""
echo "      Automated trigger setup requires connecting a GitHub/GitLab repo."
echo "      Do this in the GCP Console (one-time UI step):"
echo ""
echo "      1. Go to: https://console.cloud.google.com/cloud-build/triggers"
echo "      2. Click 'Connect Repository' → choose GitHub"
echo "      3. Select your repo → click 'Create Trigger'"
echo "      4. Set config file: cloudbuild.yaml"
echo "      5. Add substitution: _REGION=$REGION"
echo ""
echo "      Or trigger a manual build right now with:"
echo "        gcloud builds submit --config=cloudbuild.yaml \\"
echo "          --substitutions=_REGION=$REGION,_SERVICE_NAME=$SERVICE_NAME,_REPOSITORY=$REPOSITORY"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "  1. Upload documents:"
echo "     gsutil -m cp data/raw/* gs://$BUCKET_NAME/docs/"
echo ""
echo "  2. Build the index and upload vector store:"
echo "     python -m pipeline.build_index gs://$BUCKET_NAME/docs/"
echo "     gsutil -m cp vector_store/* gs://$BUCKET_NAME/vector_store/"
echo ""
echo "  3. Run first deployment:"
echo "     gcloud builds submit --config=cloudbuild.yaml \\"
echo "       --substitutions=_REGION=$REGION,_SERVICE_NAME=$SERVICE_NAME,_REPOSITORY=$REPOSITORY"
echo ""
echo "  4. Set GCS vector store URI on Cloud Run:"
echo "     gcloud run services update $SERVICE_NAME \\"
echo "       --update-env-vars GCS_VECTOR_STORE_URI=gs://$BUCKET_NAME/vector_store/ \\"
echo "       --region $REGION"
echo "=================================================="
