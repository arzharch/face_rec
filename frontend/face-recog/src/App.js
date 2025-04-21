import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";

const App = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setImage(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:8001/predict", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        setResult(data);
        if (data.error) {
          setError(data.error);
        }
      })
      .catch((err) => {
        console.error(err);
        setError("Failed to connect to backend.");
      })
      .finally(() => setIsLoading(false));
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: "image/*",
    maxFiles: 1,
  });

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🎬 Hollywood Face Recognition</h1>
      <p style={styles.subtitle}>Upload a photo to identify actors and their works</p>

      <div {...getRootProps()} style={styles.dropzone}>
        <input {...getInputProps()} />
        <p>📤 Drag & drop a celebrity photo here, or click to select</p>
      </div>

      {isLoading && (
        <div style={styles.loading}>
          <div className="spinner"></div>
          <p>Analyzing image...</p>
        </div>
      )}

      {image && (
        <div style={styles.resultsContainer}>
          <div style={styles.imageContainer}>
            <img src={image} alt="Uploaded" style={styles.image} />
            {result?.actor && (
              <h2 style={styles.actorName}>
                {result.actor}
                {result.confidence !== undefined && (
                  <span style={styles.confidenceBadge}>
                    {result.confidence.toFixed(0)}% match
                  </span>
                )}
              </h2>
            )}
          </div>

          {error && <p style={styles.error}>❌ {error}</p>}

          {result?.movies?.length > 0 && (
            <div style={styles.moviesContainer}>
              <h3 style={styles.moviesTitle}>Filmography</h3>
              <div className="grid-container">
                {result.movies.map((item, index) => (
                  <div className="card" key={index}>
                    <div className="card-inner">
                      <div className="card-front">
                        {item.poster_path ? (
                          <img
                            src={item.poster_path}
                            alt={item.title}
                            onError={(e) => {
                              e.target.src = "https://via.placeholder.com/300x450?text=No+Poster";
                            }}
                          />
                        ) : (
                          <div className="no-poster">
                            <span>No Image Available</span>
                          </div>
                        )}
                        <div className="card-title">
                          <h4>{item.title}</h4>
                          <p className="media-type">{item.media_type.toUpperCase()}</p>
                        </div>
                      </div>
                      <div className="card-back">
                        <h4>{item.title}</h4>
                        <p className="overview">
                          {item.overview || "No overview available."}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    textAlign: "center",
    padding: "20px",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    maxWidth: "1200px",
    margin: "0 auto",
  },
  title: {
    fontSize: "2.5rem",
    marginBottom: "0.5rem",
    color: "#333",
  },
  subtitle: {
    fontSize: "1.1rem",
    color: "#666",
    marginBottom: "2rem",
  },
  dropzone: {
    border: "2px dashed #aaa",
    padding: "30px",
    borderRadius: "12px",
    backgroundColor: "#f9f9f9",
    cursor: "pointer",
    marginBottom: "20px",
    transition: "all 0.3s ease",
  },
  dropzoneHover: {
    borderColor: "#4CAF50",
    backgroundColor: "#f0fff0",
  },
  loading: {
    margin: "20px 0",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "10px",
  },
  resultsContainer: {
    marginTop: "30px",
    textAlign: "left",
  },
  imageContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    marginBottom: "30px",
  },
  image: {
    maxWidth: "300px",
    maxHeight: "300px",
    borderRadius: "10px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    objectFit: "cover",
  },
  actorName: {
    marginTop: "15px",
    fontSize: "1.8rem",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  confidenceBadge: {
    fontSize: "1rem",
    backgroundColor: "#4CAF50",
    color: "white",
    padding: "3px 10px",
    borderRadius: "20px",
  },
  moviesContainer: {
    marginTop: "20px",
  },
  moviesTitle: {
    fontSize: "1.5rem",
    marginBottom: "20px",
    paddingBottom: "10px",
    borderBottom: "1px solid #eee",
  },
  error: {
    color: "#d32f2f",
    backgroundColor: "#fdecea",
    padding: "10px",
    borderRadius: "4px",
    margin: "20px 0",
  },
};

export default App;