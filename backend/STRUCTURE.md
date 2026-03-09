# Backend File Structure

## 📁 Directory Overview

```
backend/
├── 🌐 API (Production)
│   ├── src/                      # Organized API code
│   │   ├── api/                  # API layer
│   │   │   ├── routes/           # HTTP endpoints
│   │   │   └── controllers/      # Business logic
│   │   ├── core/                 # Core functionality
│   │   │   ├── config.py         # Configuration
│   │   │   └── model_loader.py   # Model management
│   │   └── utils/                # Utilities
│   │       └── image_preprocessing.py
│   └── server.py                 # FastAPI entry point ⚡
│
├── 🔬 Training & Evaluation
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Model evaluation
│   ├── build_model.py            # Model architectures
│   ├── load_data.py              # Dataset loaders
│   └── preprocess_data.py        # Data preprocessing
│
├── 💾 Data & Models
│   ├── models/                   # Production models (Git tracked)
│   └── runs/                     # Training experiments (Gitignored)
│
├── ⚙️ Configuration
│   ├── .env.example              # Environment template
│   ├── .gitignore                # Git ignore rules
│   └── requirements.txt          # Python dependencies
│
├── 📚 Documentation
│   ├── README.md                 # Full documentation
│   ├── QUICKSTART.md             # Quick reference
│   └── STRUCTURE.md              # This file
│
└── 🔧 Environment
    └── venv/                     # Python virtual environment
```

## 📝 File Descriptions

### API Files (Production)

| File | Purpose |
|------|---------|
| `server.py` | FastAPI application entry point. Run this to start the API. |
| `src/api/routes/predict.py` | Prediction endpoints (`/api/predict`) |
| `src/api/controllers/prediction_controller.py` | Prediction business logic |
| `src/core/model_loader.py` | Loads and caches TensorFlow models |
| `src/core/config.py` | Configuration management (reads `.env`) |
| `src/utils/image_preprocessing.py` | Image preprocessing utilities |

### Training Scripts

| File | Purpose |
|------|---------|
| `train.py` | **Main training script** - trains models and saves to `models/` |
| `evaluate.py` | Evaluates trained models, generates confusion matrices |
| `build_model.py` | Defines CNN architectures (lite/standard/large) |
| `load_data.py` | Loads datasets via TensorFlow Datasets |
| `preprocess_data.py` | Data normalization, augmentation, batching |

### Data & Models

| Path | Purpose |
|------|---------|
| `models/*.keras` | Production-ready trained models (Git tracked) |
| `models/*_metadata.json` | Model performance metrics |
| `models/*_config.json` | Training configurations |
| `runs/` | Training experiments (Gitignored - too large) |

## 🚀 Common Commands

### Start API Server
```powershell
python server.py
# or
uvicorn server:app --reload
```

### Train a Model
```powershell
python train.py --dataset emnist_digits --epochs 10
```

### Evaluate a Model
```powershell
python evaluate.py --model_path models/mymodel_v1.keras --dataset emnist_digits
```

## 🧹 What Was Removed

The following obsolete files were removed during cleanup:
- ❌ `predict.py` - Old CLI prediction script (now handled by API)
- ❌ `__pycache__/` - Python cache (regenerated automatically)

## 📊 Size Guidelines

- **API code** (`src/`): ~50 KB
- **Training scripts**: ~200 KB  
- **Models** (`models/`): ~1-20 MB per model (Git tracked)
- **Training runs** (`runs/`): Can be GBs (Gitignored)
- **Virtual env** (`venv/`): ~500 MB (Gitignored)

## 🤝 Contributing

When adding new features:
- **API endpoints** → `src/api/routes/`
- **Business logic** → `src/api/controllers/`
- **Core utilities** → `src/core/` or `src/utils/`
- **Training changes** → Root-level `.py` files
- **New datasets** → Add to `load_data.py`
- **New architectures** → Add to `build_model.py`

## 📦 For Deployment

Files to include in Git:
- ✅ All API code (`src/`, `server.py`)
- ✅ Training scripts (root `.py` files)
- ✅ Configuration (`.env.example`, `.gitignore`, `requirements.txt`)
- ✅ Documentation (`.md` files)
- ✅ Production models (`models/`)

Files to exclude (already in `.gitignore`):
- ❌ Virtual environment (`venv/`)
- ❌ Training runs (`runs/`)
- ❌ Python cache (`__pycache__/`)
- ❌ Environment secrets (`.env`)
- ❌ IDE files (`.vscode/`, `.idea/`)
