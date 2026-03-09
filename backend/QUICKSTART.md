# 🚀 Quick Start Guide

## Start Development Server (5 seconds)

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python server.py
```

Then open: http://127.0.0.1:8000/docs

## Train a New Model (Quick)

```powershell
python train.py --dataset emnist_digits --epochs 10 --batch_size 128
```

Model will be saved to `models/` folder automatically.

## Test Prediction with cURL

```powershell
curl -X POST "http://127.0.0.1:8000/api/predict" `
  -F "file=@path\to\your\image.png"
```

## Key Endpoints

- **API Docs:** http://127.0.0.1:8000/docs
- **Predict:** `POST /api/predict` (upload image file)
- **Model Info:** `GET /api/predict/model-info`
- **Health:** `GET /health`

## Frontend Integration

```javascript
// Upload image for prediction
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://127.0.0.1:8000/api/predict?top_k=5', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log(result.top_prediction);  // e.g., "5"
console.log(result.confidence);      // e.g., 0.987
```

## Change Model

Edit `src/core/config.py`:

```python
model_path: str = "models/your_model_v1.keras"
dataset_name: str = "emnist_byclass"
```

Or set environment variables in `.env`:

```env
MODEL_PATH=models/your_model_v1.keras
DATASET_NAME=emnist_byclass
```

## Common Issues

**Port already in use:**
```powershell
# Use different port
uvicorn server:app --port 8001
```

**Model not found:**
- Check `models/` folder contains `.keras` file
- Update `MODEL_PATH` in config or `.env`

**CORS errors:**
- Add your frontend URL to `CORS_ORIGINS` in `.env`

For full documentation, see [README.md](README.md)
