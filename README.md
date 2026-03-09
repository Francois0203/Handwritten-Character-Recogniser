# Handwritten Character Recognition - Full Stack Application

A complete full-stack application for handwritten character recognition with a FastAPI backend and React frontend.

## Project Structure

```
Handwritten-Character-Recogniser/
├── backend/              # FastAPI backend server
│   ├── server.py         # FastAPI application entry point
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Model evaluation script
│   ├── models/           # Production models (version-controlled)
│   ├── src/              # Application source code
│   │   ├── api/          # API routes and controllers
│   │   │   ├── routes/   # API endpoints
│   │   │   └── controllers/  # Business logic
│   │   ├── core/         # Core functionality (config, model loader)
│   │   └── utils/        # Utility functions (image preprocessing)
│   ├── venv/             # Python virtual environment (not tracked)
│   └── runs/             # Training runs and logs (not tracked)
│
└── frontend/             # React frontend application
    ├── src/
    │   ├── App.jsx       # Main application component
    │   ├── App.css       # Application styles
    │   └── main.jsx      # Entry point
    ├── public/           # Static assets
    └── dist/             # Production build (not tracked)
```

## Quick Start

### 1. Start the Backend

```bash
# Navigate to backend
cd backend

# Activate virtual environment
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/Mac

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the server
uvicorn server:app --reload
```

Backend will run at: **http://127.0.0.1:8000**

### 2. Start the Frontend

In a new terminal:

```bash
# Navigate to frontend
cd frontend

# Install dependencies (if not already installed)
npm install

# Start development server
npm run dev
```

Frontend will run at: **http://localhost:5173**

### 3. Test the Application

1. Open your browser to http://localhost:5173
2. Upload an image of a handwritten character
3. Click "Predict Character"
4. See the prediction results with confidence scores!

## API Endpoints

### Health Check
```
GET http://127.0.0.1:8000/
```

### Model Information
```
GET http://127.0.0.1:8000/api/predict/model-info
```

### Predict Character
```
POST http://127.0.0.1:8000/api/predict?top_k=5
Content-Type: multipart/form-data
Body: file (image)
```

## Training & Evaluation

### Train a New Model

```bash
cd backend
venv\Scripts\activate
python train.py
```

Configuration options in `train.py`:
- Dataset: `emnist_digits`, `emnist_letters`, `emnist_balanced`
- Model size: `lite`, `standard`, `large`
- Learning rate, batch size, epochs, etc.

Best models are automatically saved to `backend/models/` for deployment.

### Evaluate Model

```bash
python evaluate.py
```

Generates:
- Confusion matrices (confusion_matrix.png, confusion_matrix_normalized.png)
- Per-class performance report (classification_report.txt)
- Metrics summary (metrics.json)

## Deployment

### Backend Deployment

The backend is ready to deploy:
- All models in `backend/models/` are version-controlled
- Environment configuration via `.env` file
- CORS configured for frontend origin
- Production-ready structure with FastAPI

### Frontend Deployment

Build the frontend:

```bash
cd frontend
npm run build
```

The `dist/` folder contains production-ready static files.

Options:
- Deploy to Vercel, Netlify, or any static hosting
- Serve with backend (copy dist/ to backend/static/)
- Update API URL in production build

## Technology Stack

### Backend
- **Python 3.10+**
- **TensorFlow/Keras 2.20+** - Deep learning model
- **FastAPI 0.110+** - REST API framework
- **Uvicorn** - ASGI server
- **Pydantic-settings** - Configuration management
- **Pillow** - Image processing

### Frontend
- **React** - UI framework
- **Vite 7.3+** - Build tool
- **Modern CSS** - Gradient UI with animations

### Model
- **EMNIST Datasets** - Handwritten characters
- **CNN Architecture** - Custom residual blocks
- **WarmupCosineDecay LR Schedule** - Training optimization

## Git & Version Control

Ready to push to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: Full-stack handwritten character recognition"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Files tracked:
- ✅ Source code (frontend/, backend/src/, backend/train.py, etc.)
- ✅ Production models (backend/models/)
- ✅ Configuration files (requirements.txt, package.json, etc.)
- ✅ Documentation (README files)

Files ignored (.gitignore):
- ❌ Virtual environment (backend/venv/)
- ❌ Training runs (backend/runs/)
- ❌ Python cache (\_\_pycache\_\_/)
- ❌ Node modules (frontend/node_modules/)
- ❌ Build outputs (frontend/dist/)
- ❌ Environment variables (.env)

## Support

For more details:
- Backend: See `backend/README.md`, `backend/QUICKSTART.md`, `backend/STRUCTURE.md`
- Frontend: See `frontend/README.md`

## Model Performance

Current model (emnist_digits_standard_v1):
- **Accuracy**: 99.68%
- **Parameters**: 768,170
- **Classes**: 10 (digits 0-9)

Trained on EMNIST dataset with data augmentation and learning rate warmup/cosine decay.
