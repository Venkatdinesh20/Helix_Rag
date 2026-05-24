<#
.SYNOPSIS
  One-time GCP infrastructure setup for the Helix RAG pipeline (Windows / PowerShell).

.DESCRIPTION
  Mirror of setup_gcp.sh for Windows users. Safe to re-run — every gcloud
  command is idempotent.

  Creates / configures:
    * Required APIs (Run, Cloud Build, Artifact Registry, Secret Manager, Storage)
    * Artifact Registry Docker repository
    * Cloud Storage bucket for documents + vector store
    * Secret Manager secret holding the OpenAI API key
    * IAM bindings for the Cloud Build service account

.PREREQUISITES
  1. gcloud CLI installed     →  https://cloud.google.com/sdk/docs/install
  2. Logged in                →  gcloud auth login
  3. Project selected         →  gcloud config set project YOUR_PROJECT_ID
  4. Billing enabled on the project (Cloud Run requires it, even on free tier).

.USAGE
  PS> .\setup_gcp.ps1
  PS> .\setup_gcp.ps1 -Region us-central1 -ServiceName rag-api
#>

[CmdletBinding()]
param(
  [string]$Region        = 'us-central1',   # free-tier-eligible
  [string]$ServiceName   = 'rag-api',
  [string]$Repository    = 'rag-repo',
  [string]$BucketSuffix  = 'rag-data',
  [string]$SecretName    = 'openai-api-key'
)

$ErrorActionPreference = 'Stop'

function Write-Step($n, $msg) { Write-Host ""; Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)       { Write-Host "      $msg" -ForegroundColor Green }
function Write-Skip($msg)     { Write-Host "      $msg" -ForegroundColor DarkGray }

# ── Verify gcloud is on PATH ────────────────────────────────────────────────
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
  Write-Host "ERROR: gcloud CLI not found on PATH." -ForegroundColor Red
  Write-Host "Install from: https://cloud.google.com/sdk/docs/install#windows"
  exit 1
}

$projectId = (gcloud config get-value project 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($projectId) -or $projectId -eq '(unset)') {
  Write-Host "ERROR: No GCP project selected." -ForegroundColor Red
  Write-Host "Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
}
$bucketName = "$projectId-$BucketSuffix"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Yellow
Write-Host " Helix RAG — GCP Infrastructure Setup"               -ForegroundColor Yellow
Write-Host " Project : $projectId"
Write-Host " Region  : $Region"
Write-Host " Service : $ServiceName"
Write-Host "==================================================" -ForegroundColor Yellow

# ── Step 1: Enable APIs ──────────────────────────────────────────────────────
Write-Step '1/6' 'Enabling GCP APIs…'
gcloud services enable `
  run.googleapis.com `
  cloudbuild.googleapis.com `
  artifactregistry.googleapis.com `
  secretmanager.googleapis.com `
  storage.googleapis.com `
  --project=$projectId | Out-Null
Write-Ok 'APIs enabled.'

# ── Step 2: Artifact Registry ────────────────────────────────────────────────
Write-Step '2/6' "Creating Artifact Registry repository: $Repository …"
$exists = gcloud artifacts repositories describe $Repository --location=$Region --project=$projectId 2>$null
if ($LASTEXITCODE -eq 0) {
  Write-Skip '(already exists — skipping)'
} else {
  gcloud artifacts repositories create $Repository `
    --repository-format=docker `
    --location=$Region `
    --description="Helix RAG Docker images" `
    --project=$projectId | Out-Null
  Write-Ok "Repository created."
}

# ── Step 3: GCS bucket ───────────────────────────────────────────────────────
Write-Step '3/6' "Creating Cloud Storage bucket: gs://$bucketName …"
gcloud storage buckets describe "gs://$bucketName" --project=$projectId 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Skip '(already exists — skipping)'
} else {
  gcloud storage buckets create "gs://$bucketName" `
    --location=$Region `
    --project=$projectId | Out-Null
  Write-Ok "Bucket created."
}
Write-Ok "Upload docs  : gcloud storage cp data/raw/* gs://$bucketName/docs/"
Write-Ok "Upload index : gcloud storage cp vector_store/* gs://$bucketName/vector_store/"

# ── Step 4: Secret Manager — OpenAI API key ─────────────────────────────────
Write-Step '4/6' "Creating Secret Manager secret: $SecretName …"
gcloud secrets describe $SecretName --project=$projectId 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Skip '(already exists — skipping)'
  Write-Skip "To rotate: `$key='sk-...'; `$key | gcloud secrets versions add $SecretName --data-file=-"
} else {
  $key = Read-Host -Prompt 'Paste your OpenAI API key (input hidden)' -AsSecureString
  $plain = [System.Net.NetworkCredential]::new('', $key).Password
  $tmp = New-TemporaryFile
  try {
    # -NoNewline matters — Secret Manager stores bytes verbatim, a trailing
    # newline will break OpenAI auth.
    [System.IO.File]::WriteAllText($tmp.FullName, $plain)
    gcloud secrets create $SecretName --data-file=$tmp.FullName --project=$projectId | Out-Null
    Write-Ok 'Secret created.'
  } finally {
    Remove-Item $tmp.FullName -Force -ErrorAction SilentlyContinue
  }
}

# ── Step 5: IAM bindings for Cloud Build ─────────────────────────────────────
Write-Step '5/6' 'Granting IAM roles to Cloud Build service account…'
$projectNumber = (gcloud projects describe $projectId --format='value(projectNumber)').Trim()
# Modern Cloud Build uses the project default compute SA; legacy projects use the
# @cloudbuild.gserviceaccount.com SA. We grant both to be robust.
$cbSAs = @(
  "$projectNumber@cloudbuild.gserviceaccount.com",
  "$projectNumber-compute@developer.gserviceaccount.com"
)
$roles = @(
  'roles/run.admin',
  'roles/iam.serviceAccountUser',
  'roles/secretmanager.secretAccessor',
  'roles/storage.objectViewer',
  'roles/artifactregistry.writer'
)
foreach ($sa in $cbSAs) {
  foreach ($role in $roles) {
    gcloud projects add-iam-policy-binding $projectId `
      --member="serviceAccount:$sa" `
      --role=$role --quiet 2>$null | Out-Null
  }
}
Write-Ok 'IAM roles granted.'

# ── Step 6: Trigger guidance ─────────────────────────────────────────────────
Write-Step '6/6' 'Cloud Build trigger (one-time UI step)'
Write-Host ""
Write-Host "  Choose ONE deployment mode:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  A) Manual one-shot deploy from this machine:"
Write-Host "       gcloud builds submit --config cloudbuild.yaml --region=$Region ."
Write-Host ""
Write-Host "  B) Automatic CI/CD on every push to main:"
Write-Host "       1. Push this repo to GitHub"
Write-Host "       2. Open https://console.cloud.google.com/cloud-build/triggers?project=$projectId"
Write-Host "       3. Click 'Connect Repository' → choose GitHub → select repo"
Write-Host "       4. Create trigger → config file: cloudbuild.yaml → branch: ^main$"
Write-Host ""
Write-Host "==================================================" -ForegroundColor Yellow
Write-Host " Setup complete." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Yellow
