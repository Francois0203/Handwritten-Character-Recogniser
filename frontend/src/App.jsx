import { useState } from 'react';
import './App.css';

const API_URL = 'http://127.0.0.1:8000/api/predict';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPredictions(null);
      setError(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch(`${API_URL}?top_k=5`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        setPredictions(result);
      } else {
        setError(result.error || 'Prediction failed');
      }
    } catch (err) {
      setError(`Failed to connect to backend: ${err.message}`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setPredictions(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>✍️ Handwritten Character Recognition</h1>
        <p>Upload an image to recognize handwritten characters</p>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="upload-area">
            {!imagePreview ? (
              <label className="upload-label">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  className="file-input"
                />
                <div className="upload-prompt">
                  <svg
                    className="upload-icon"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <span className="upload-text">Click to upload image</span>
                  <span className="upload-hint">PNG, JPG, BMP up to 10MB</span>
                </div>
              </label>
            ) : (
              <div className="image-preview">
                <img src={imagePreview} alt="Preview" className="preview-image" />
                <button onClick={handleReset} className="reset-button">
                  ✕ Remove
                </button>
              </div>
            )}
          </div>

          {selectedImage && !predictions && (
            <button
              onClick={handlePredict}
              disabled={loading}
              className="predict-button"
            >
              {loading ? (
                <>
                  <span className="loading-spinner"></span>
                  Analyzing...
                </>
              ) : (
                <>🔍 Predict Character</>
              )}
            </button>
          )}
        </div>

        {error && (
          <div className="error-message">
            <strong>⚠️ Error:</strong> {error}
          </div>
        )}

        {predictions && (
          <div className="results-section">
            <h2 className="results-title">Predictions</h2>

            <div className="top-prediction">
              <div className="prediction-label">Top Prediction</div>
              <div className="prediction-character">
                {predictions.top_prediction}
              </div>
              <div className="prediction-confidence">
                {(predictions.confidence * 100).toFixed(1)}% confident
              </div>
            </div>

            <div className="all-predictions">
              <h3 className="predictions-subtitle">All Predictions</h3>
              {predictions.predictions.map((pred) => (
                <div key={pred.rank} className="prediction-item">
                  <div className="prediction-rank">#{pred.rank}</div>
                  <div className="prediction-char">{pred.label}</div>
                  <div className="prediction-bar-container">
                    <div
                      className="prediction-bar"
                      style={{ width: `${pred.confidence * 100}%` }}
                    ></div>
                  </div>
                  <div className="prediction-percent">
                    {(pred.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>

            <button onClick={handleReset} className="try-another-button">
              Try Another Image
            </button>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          Powered by TensorFlow • FastAPI • React
        </p>
      </footer>
    </div>
  );
}

export default App;
