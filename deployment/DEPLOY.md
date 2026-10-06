# Deployment Guide

Step-by-step instructions for deploying the House Price Prediction application.

---

## Part 1 — Deploy FastAPI Backend on Render

### Prerequisites
- GitHub account with this repo pushed to a public or private repository
- [Render](https://render.com) account (free tier is sufficient)

### Step-by-Step

**1. Push the repository to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/house-price-prediction.git
git push -u origin main
```

**2. Log in to Render**

Go to <https://render.com> and sign in.

**3. Create a New Web Service**

- Click **New +** → **Web Service**
- Connect your GitHub account and select your repository
- Choose the branch (`main`)

**4. Configure the Web Service**

| Setting | Value |
|---------|-------|
| Name | `house-price-api` (or any name) |
| Region | Oregon (US West) or nearest |
| Branch | `main` |
| Root Directory | *(leave blank)* |
| Runtime | `Python 3` |
| Build Command | `pip install -r requirements.txt && python model/train.py` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free |

**5. Add Environment Variables (optional)**

No secrets are required. The `PORT` variable is automatically injected by Render.

**6. Deploy**

Click **Create Web Service**. Render will:
1. Install Python dependencies
2. Run `python model/train.py` (trains and saves model artifacts)
3. Start the FastAPI server

**7. Get Your Public URL**

After deployment, Render shows a URL like:
```
https://house-price-api-xxxx.onrender.com
```

Test it:
```bash
curl https://house-price-api-xxxx.onrender.com/health
```

Expected response:
```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

**8. Test All Endpoints**

```bash
BASE=https://house-price-api-xxxx.onrender.com

# Health
curl $BASE/health

# Feature info
curl $BASE/feature-info

# Metrics
curl $BASE/metrics

# Model info
curl $BASE/model-info

# Predict
curl -X POST $BASE/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc":3.87,"HouseAge":28,"AveRooms":5.43,"AveBedrms":1.1,"Population":1425,"AveOccup":3.07,"Latitude":35.63,"Longitude":-119.57}'
```

---

## Part 2 — Deploy Streamlit Frontend on Streamlit Community Cloud

### Prerequisites
- GitHub account with this repo
- [Streamlit Community Cloud](https://streamlit.io/cloud) account (free)
- FastAPI backend deployed on Render (Part 1)

### Step-by-Step

**1. Go to Streamlit Community Cloud**

Visit <https://streamlit.io/cloud> and sign in with GitHub.

**2. Create a New App**

- Click **New app**
- Select your GitHub repository
- Select branch: `main`
- Set **Main file path**: `streamlit_app.py`

**3. Set the Backend URL**

In the app settings, click **Advanced settings** → **Secrets** and add:

```toml
API_BASE = "https://house-price-api-xxxx.onrender.com"
```

Replace `house-price-api-xxxx` with your actual Render service name.

**4. Deploy**

Click **Deploy!**. Streamlit will install dependencies from `requirements.txt` and launch the app.

**5. Your App URL**

```
https://your-app-name.streamlit.app
```

---

## Part 3 — Docker Compose (Local or VPS)

```bash
# 1. Train the model first (required for Docker build)
python model/train.py

# 2. Build and start both containers
docker-compose up --build -d

# 3. Check logs
docker-compose logs -f

# 4. Access
#   Backend:  http://localhost:8000
#   Frontend: http://localhost:8501

# 5. Stop
docker-compose down
```

---

## Environment Variables Reference

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `PORT` | Backend | `8000` | Injected by Render automatically |
| `API_BASE` | Frontend | `http://localhost:8000` | Full URL of the FastAPI backend |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails with "model.pkl not found" | The build command includes `python model/train.py` – ensure it ran |
| Render shows "No open ports" | Verify start command uses `--host 0.0.0.0 --port $PORT` |
| Streamlit shows "Cannot reach backend" | Set `API_BASE` secret in Streamlit Cloud settings |
| Free Render service is slow | Free tier spins down after inactivity; first request may take 30-60s |
| CORS errors in browser | Backend includes `allow_origins=["*"]`; should not occur |
