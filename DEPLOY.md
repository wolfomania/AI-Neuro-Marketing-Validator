# Deploy to Modal.com — Step-by-Step Guide

This guide gets the AI Neuro-Marketing Validator running on [Modal.com](https://modal.com) with copy-paste commands. Total time: ~10 minutes.

---

## Prerequisites

- Python 3.11+ installed locally
- Node.js 18+ installed locally
- A Modal.com account (free tier: $30/month credits)
- (Optional) Anthropic API key for Claude analysis
- (Optional) HuggingFace token for TRIBE v2 model

---

## Step 1: Install Modal CLI

```bash
pip install modal
```

## Step 2: Authenticate with Modal

```bash
modal token new
```

This opens your browser. Log in and the token is saved locally.

## Step 3: Create secrets on Modal

Choose one of the options below depending on your access control needs.

### Option A: Open access (no restrictions)

```bash
modal secret create neuromarketing-secrets \
    NM_ANTHROPIC_API_KEY="" \
    NM_HF_TOKEN="" \
    NM_USE_MOCK="true" \
    NM_UPLOAD_DIR="/data/uploads" \
    NM_RESULTS_DIR="/data/results" \
    NM_CORS_ORIGINS="*"
```

### Option B: Token-protected access (recommended)

Generate a random token and save it somewhere safe:

```bash
# Generate a token (copy the output — you'll need it later)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Create the secret with the token:

```bash
modal secret create neuromarketing-secrets \
    NM_ANTHROPIC_API_KEY="" \
    NM_HF_TOKEN="" \
    NM_USE_MOCK="true" \
    NM_UPLOAD_DIR="/data/uploads" \
    NM_RESULTS_DIR="/data/results" \
    NM_CORS_ORIGINS="*" \
    NM_ACCESS_TOKEN="<paste-your-generated-token-here>"
```

### Option C: IP-restricted access

Find your public IP first:

```bash
curl -s https://ifconfig.me
```

Then create the secret with your IP(s):

```bash
modal secret create neuromarketing-secrets \
    NM_ANTHROPIC_API_KEY="" \
    NM_HF_TOKEN="" \
    NM_USE_MOCK="true" \
    NM_UPLOAD_DIR="/data/uploads" \
    NM_RESULTS_DIR="/data/results" \
    NM_CORS_ORIGINS="*" \
    NM_ALLOWED_IPS="<your-ip-here>"
```

### Option D: Both token + IP restriction

```bash
modal secret create neuromarketing-secrets \
    NM_ANTHROPIC_API_KEY="" \
    NM_HF_TOKEN="" \
    NM_USE_MOCK="true" \
    NM_UPLOAD_DIR="/data/uploads" \
    NM_RESULTS_DIR="/data/results" \
    NM_CORS_ORIGINS="*" \
    NM_ACCESS_TOKEN="<your-token>" \
    NM_ALLOWED_IPS="<your-ip>"
```

> To update secrets later: `modal secret create neuromarketing-secrets --force ...`

## Step 4: Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

Verify `frontend/dist/index.html` exists:

```bash
ls frontend/dist/index.html
```

## Step 5: Test locally with Modal dev server

```bash
modal serve modal_app.py
```

Modal prints a temporary URL like:

```
https://<workspace>--neuromarketing-webservice-web-dev.modal.run
```

Open it in your browser. The React app loads with the FastAPI backend.

Press `Ctrl+C` to stop.

## Step 6: Deploy to production

```bash
modal deploy modal_app.py
```

Modal prints a permanent URL:

```
https://<workspace>--neuromarketing-webservice-web.modal.run
```

This URL is live. The service auto-scales to zero when idle (no cost).

---

## Access Control

### How token auth works

When `NM_ACCESS_TOKEN` is set, every API request must include:

```
Authorization: Bearer <your-token>
```

**For the frontend**, open the browser console on your deployed app and run:

```javascript
localStorage.setItem('nm_access_token', '<your-token>')
```

Then refresh the page. The token persists in your browser across sessions.

To revoke access, change the token in Modal secrets and redeploy:

```bash
modal secret create neuromarketing-secrets --force \
    NM_ACCESS_TOKEN="<new-token>" \
    ... (rest of your vars)
modal deploy modal_app.py
```

### How IP restriction works

When `NM_ALLOWED_IPS` is set, only requests from those IPs are allowed. Multiple IPs are comma-separated:

```
NM_ALLOWED_IPS="203.0.113.10,198.51.100.42"
```

The `/api/health` endpoint is always accessible (no auth required).

### Testing access control

```bash
# Should return 401 (no token)
curl https://<your-url>/api/health

# Should return 200 (health is always open)
curl https://<your-url>/api/health

# With token
curl -H "Authorization: Bearer <your-token>" https://<your-url>/api/analysis
```

---

## Enabling Real AI Analysis (API keys)

To use Claude analysis instead of mock mode, update the secret:

```bash
modal secret create neuromarketing-secrets --force \
    NM_ANTHROPIC_API_KEY="sk-ant-your-key-here" \
    NM_HF_TOKEN="hf_your-token-here" \
    NM_USE_MOCK="false" \
    NM_UPLOAD_DIR="/data/uploads" \
    NM_RESULTS_DIR="/data/results" \
    NM_CORS_ORIGINS="*" \
    NM_ACCESS_TOKEN="<your-token>"
```

Then redeploy:

```bash
modal deploy modal_app.py
```

> For GPU inference (real TRIBE v2), the `GpuService` class in `modal_app.py` uses an A10G GPU. To switch to GPU mode, update the Modal deployment to use `GpuService` instead of `WebService`.

---

## Cost Estimate

| Scenario | Monthly Cost |
|----------|-------------|
| Development / testing (free tier) | $0 (within $30 credit) |
| Light usage, scale-to-zero | ~$1-5 |
| Always-warm CPU container (24/7) | ~$35 |
| Always-warm A10G GPU (24/7) | ~$790 |

Modal bills per-second. Idle containers (scale-to-zero) cost nothing.

---

## Useful Commands

```bash
# View logs
modal app logs neuromarketing

# List deployments
modal app list

# Stop a deployment
modal app stop neuromarketing

# Update secrets (requires --force to overwrite)
modal secret create neuromarketing-secrets --force KEY=value ...

# Run one-off function
modal run modal_app.py::download_model
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `modal: command not found` | Run `pip install modal` again |
| `Secret not found` | Check name matches: `modal secret list` |
| Frontend shows blank page | Rebuild: `cd frontend && npm run build` |
| 401 on every request | Set token in browser: `localStorage.setItem('nm_access_token', '...')` |
| 403 IP not allowed | Check your IP: `curl ifconfig.me` and update `NM_ALLOWED_IPS` |
| Cold start takes 30s+ | Set `min_containers=1` in `modal_app.py` (costs ~$0.05/hr for CPU) |
