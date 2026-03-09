# Handwritten Character Recognition - Frontend

A React-based web interface for handwritten character recognition using the backend API.

## Quick Start

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Start Development Server**:
   ```bash
   npm run dev
   ```

   The app will be available at: http://localhost:5173/

3. **Make sure the Backend is Running**:
   The frontend expects the backend API to be running at `http://127.0.0.1:8000`
   
   To start the backend:
   ```bash
   cd ../backend
   venv\Scripts\activate
   uvicorn server:app --reload
   ```

## Features

- **Image Upload**: Upload handwritten character images (PNG, JPG, BMP)
- **Image Preview**: See the uploaded image before prediction
- **Character Prediction**: Get instant predictions from the trained model
- **Confidence Display**: View top 5 predictions with confidence scores
- **Visual Results**: Beautiful gradient UI with confidence bars

## API Connection

The frontend connects to the backend API at:
- **Prediction Endpoint**: `POST http://127.0.0.1:8000/api/predict?top_k=5`
- **CORS**: Already configured in backend for `http://localhost:5173`

## Build for Production

```bash
npm run build
```

The production-ready files will be in the `dist/` folder.

## Technologies

- **React** - UI framework
- **Vite** - Build tool and dev server
- **CSS3** - Styling with gradients and animations
