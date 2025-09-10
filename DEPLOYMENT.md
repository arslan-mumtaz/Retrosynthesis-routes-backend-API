# AlphaReact API Deployment Guide

## 🚀 Railway Deployment

### Quick Deploy
1. Push your code to GitHub
2. Connect your GitHub repo to Railway
3. Set environment variables in Railway Dashboard (see below)
4. Railway will automatically detect the `Dockerfile` and deploy

### Required Environment Variables
Set these in Railway Dashboard:
```
PORT=8000
HOST=0.0.0.0
PYTHONPATH=/app
ENVIRONMENT=production

# Optional: Auto-download model files
ZINC_STOCK_URL=https://your-storage.com/zinc_stock.hdf5
USPTO_MODEL_URL=https://your-storage.com/uspto_model.onnx
USPTO_RINGBREAKER_MODEL_URL=https://your-storage.com/uspto_ringbreaker_model.onnx
USPTO_FILTER_MODEL_URL=https://your-storage.com/uspto_filter_model.onnx
USPTO_TEMPLATES_URL=https://your-storage.com/uspto_templates.csv.gz
USPTO_RINGBREAKER_TEMPLATES_URL=https://your-storage.com/uspto_ringbreaker_templates.csv.gz
```

### What Happens:
- **With URLs**: Files are automatically downloaded on startup
- **Without URLs**: Only forward prediction works (retrosynthesis disabled)
- **Missing files**: API still works but with limited functionality

### Local Testing
```bash
# Build Docker image
docker build -t alphareact-api .

# Run container
docker run -p 8000:8000 alphareact-api

# Test API
curl http://localhost:8000/health
```

## 📡 API Endpoints

### Base URL: `https://your-app.railway.app`

### Endpoints:
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /find_routes` - Find retrosynthesis routes

### Example Request:
```bash
curl -X POST "https://your-app.railway.app/find_routes" \
  -H "Content-Type: application/json" \
  -d '{
    "target_smiles": "CCO",
    "max_routes": 5
  }'
```

### Example Response:
```json
{
  "success": true,
  "target_smiles": "CCO",
  "total_routes_found": 3,
  "routes_returned": 3,
  "routes": [
    {
      "route_id": 1,
      "total_steps": 2,
      "target_molecule": "CCO",
      "steps": [...]
    }
  ]
}
```

## 🔧 Development

### Run Locally:
```bash
# API mode
python reaction_predictor.py --mode api

# Test mode
python reaction_predictor.py --mode test --target-smiles "CCO"
```

### Environment Setup:
```bash
pip install -r requirements.txt
```

## 📋 Features

- ✅ FastAPI with automatic docs
- ✅ Docker containerized
- ✅ Railway deployment ready
- ✅ Health checks
- ✅ Error handling
- ✅ Retrosynthesis analysis
- ✅ JSON output format
- ✅ Python 3.11 support
