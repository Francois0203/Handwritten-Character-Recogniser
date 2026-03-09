# Character Recognition Backend

A FastAPI-based REST API for handwritten character recognition using deep learning (TensorFlow/Keras).

## 🚀 Features

- **REST API** for character recognition from images
- **Multiple datasets** support (MNIST, EMNIST digits, EMNIST letters, etc.)
- **Top-k predictions** with confidence scores
- **CORS enabled** for frontend integration
- **Model hot-loading** on server startup
- **Comprehensive logging** and error handling
- **Production-ready** structure with separation of concerns

## 📁 Project Structure

```
backend/
├── src/                     # 🌐 API Code (Production)
│   ├── api/
│   │   ├── routes/          # API endpoints
│   │   │   └── predict.py   # Prediction routes
│   │   └── controllers/     # Business logic
│   │       └── prediction_controller.py
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   └── model_loader.py  # Model loading and caching
│   └── utils/
│       └── image_preprocessing.py  # Image utilities
│
├── models/                  # 💾 Production models (Git tracked)
│   ├── *.keras             # Model files
│   ├── *_metadata.json     # Model metadata
│   └── *_config.json       # Training configs
│
├── runs/                    # 🔬 Training experiments (Gitignored)
│
├── train.py                 # 🎓 Model training script
├── evaluate.py              # 📊 Model evaluation script
├── build_model.py           # 🏗️ Model architecture definitions
├── load_data.py             # 📥 Dataset loaders
├── preprocess_data.py       # ⚙️ Data preprocessing
│
├── server.py                # ⚡ FastAPI application entry point
├── requirements.txt         # 📦 Python dependencies
├── .env.example            # 🔐 Example environment variables
├── README.md               # 📖 This file
├── QUICKSTART.md           # 🚀 Quick reference
└── STRUCTURE.md            # 📁 Detailed structure guide
```

> **📁 See [STRUCTURE.md](STRUCTURE.md) for detailed file descriptions and organization guidelines.**

## 🛠️ Setup

### 1. Create and Activate Virtual Environment

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)

```powershell
# Copy example env file
Copy-Item .env.example .env

# Edit .env to customize settings (optional)
```

## 🎯 Training a Model

### Quick Start - Train a digit recognition model

```powershell
python train.py --dataset emnist_digits --epochs 10 --batch_size 128
```

### Full Training Options

```powershell
python train.py \
    --dataset emnist_byclass \
    --architecture standard \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --optimizer adam \
    --output_dir runs/my_experiment
```

**Available datasets:**
- `mnist` - Classic MNIST digits (60k train / 10k test)
- `emnist_digits` - EMNIST digits (240k train / 40k test)
- `emnist_byclass` - Digits + letters: 0-9, A-Z, a-z (62 classes)
- `emnist_letters` - Letters only (37 classes)
- `kmnist` - Japanese Hiragana characters
- `all_combined` - All datasets combined (72 classes)

**Architectures:**
- `lite` - Small CNN (~200K params) - fast, good for digits
- `standard` - Residual CNN (~1.2M params) - recommended
- `large` - Large CNN (~4M params) - maximum accuracy

### After Training

The best model is automatically copied to `models/` with metadata:
- `models/emnistdigits_standard_v1.keras` - Model file
- `models/emnistdigits_standard_v1_metadata.json` - Performance metrics
- `models/emnistdigits_standard_v1_config.json` - Training configuration

## 🚀 Running the API Server

### Start the server

```powershell
# Development mode (auto-reload on code changes)
uvicorn server:app --reload --host 127.0.0.1 --port 8000

# Production mode
python server.py
```

### Verify it's running

Open your browser to:
- **API Root:** http://127.0.0.1:8000
- **Interactive Docs:** http://127.0.0.1:8000/docs
- **Alternative Docs:** http://127.0.0.1:8000/redoc

## 📡 API Endpoints

### 1. Predict Character

**Endpoint:** `POST /api/predict`

Upload an image file to get character predictions.

**Example using curl:**

```powershell
curl -X POST "http://127.0.0.1:8000/api/predict?top_k=5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.png"
```

**Example Response:**

```json
{
  "success": true,
  "predictions": [
    {"rank": 1, "label": "5", "confidence": 0.987},
    {"rank": 2, "label": "3", "confidence": 0.008},
    {"rank": 3, "label": "6", "confidence": 0.003},
    {"rank": 4, "label": "8", "confidence": 0.001},
    {"rank": 5, "label": "9", "confidence": 0.0005}
  ],
  "top_prediction": "5",
  "confidence": 0.987
}
```

**Parameters:**
- `file` (required) - Image file (PNG, JPG, BMP, etc.)
- `top_k` (optional) - Number of predictions to return (default: 5, max: 20)

### 2. Get Model Info

**Endpoint:** `GET /api/predict/model-info`

Get information about the loaded model.

**Example:**

```powershell
curl http://127.0.0.1:8000/api/predict/model-info
```

**Response:**

```json
{
  "loaded": true,
  "dataset": "emnist_digits",
  "num_classes": 10,
  "num_parameters": 1234567,
  "input_shape": "(None, 28, 28, 1)",
  "output_shape": "(None, 10)"
}
```

### 3. Health Check

**Endpoint:** `GET /health`

Check if the server and model are ready.

```powershell
curl http://127.0.0.1:8000/health
```

## 🧪 Evaluating a Model

```powershell
python evaluate.py \
    --model_path runs/my_run/best_model.keras \
    --dataset emnist_digits \
    --output_dir runs/my_run/eval
```

Generates:
- Confusion matrices
- Per-class accuracy charts
- Worst predictions visualization
- Classification report (JSON)
- Evaluation summary

## 🖼️ Command-Line Prediction

Test a single image from the command line:

```powershell
python predict.py \
    --image path/to/image.png \
    --model models/emnist_digits_standard_v1.keras
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`) to customize:

```env
# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model settings
MODEL_PATH=models/emnist_digits_standard_v1.keras
DATASET_NAME=emnist_digits

# CORS origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Changing the Model

To use a different model:

1. Train or copy a model file to `models/`
2. Update `src/core/config.py` or set environment variables:

```env
MODEL_PATH=models/your_model_v1.keras
DATASET_NAME=emnist_byclass
```

3. Restart the server

## 🐛 Troubleshooting

### Model fails to load

**Issue:** `Could not locate class 'WarmupCosineDecay'`

**Solution:** Models are loaded with `compile=False` to avoid optimizer issues. If you still see this error, make sure `train.py` is importable (it registers the custom learning rate schedule).

### CORS errors from frontend

**Solution:** Add your frontend URL to `CORS_ORIGINS` in `.env`:

```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://yourfrontend.com
```

### Image predictions are wrong

**Common causes:**
- Image has white background → should be auto-inverted
- Image is not a single character → crop to contain only one character
- Wrong dataset → ensure `DATASET_NAME` matches training dataset

## 📦 Deployment

### For GitHub

The structure is already set up for Git:
- `models/` folder is tracked (contains production models)
- `runs/` folder is gitignored (training experiments)
- `venv/` is gitignored

```powershell
git add .
git commit -m "Add backend with trained model"
git push
```

### For Production

1. Use a production ASGI server (uvicorn with workers):

```powershell
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

2. Set environment variables for production:

```env
DEBUG=false
HOST=0.0.0.0
CORS_ORIGINS=https://yourfrontend.com
```

3. Consider using:
   - **Docker** for containerization
   - **Gunicorn + Uvicorn** for process management
   - **NGINX** as reverse proxy
   - **Git LFS** for large model files (if > 100 MB)

## 📚 Frontend Integration

### JavaScript/TypeScript Example

```typescript
async function predictCharacter(imageFile: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://127.0.0.1:8000/api/predict?top_k=5', {
    method: 'POST',
    body: formData,
  });
  
  return await response.json();
}

// Usage
const fileInput = document.querySelector<HTMLInputElement>('#imageInput');
const file = fileInput.files[0];
const result = await predictCharacter(file);
console.log('Top prediction:', result.top_prediction);
console.log('Confidence:', result.confidence);
```

### React Example

```tsx
import { useState } from 'react';

function CharacterRecognition() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict?top_k=5', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      setPredictions(result.predictions);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {loading && <p>Analyzing...</p>}
      {predictions.length > 0 && (
        <ul>
          {predictions.map((pred) => (
            <li key={pred.rank}>
              {pred.label}: {(pred.confidence * 100).toFixed(1)}%
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Create a new branch for features
2. Test thoroughly before submitting
3. Update documentation as needed

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.
