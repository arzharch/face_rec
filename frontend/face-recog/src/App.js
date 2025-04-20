import React, { useState } from "react";
import { useDropzone } from "react-dropzone";

const App = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setImage(URL.createObjectURL(file));
    setResult(null);
    setError(null);

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
      });
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Face Recognition App</h1>

      <div {...getRootProps()} style={styles.dropzone}>
        <input {...getInputProps()} />
        <p>📤 Drag & drop an image here, or click to select one</p>
      </div>

      {image && <img src={image} alt="Uploaded" style={styles.image} />}

      {error && <p style={styles.error}>❌ {error}</p>}

      {result && (
        <div style={styles.resultBox}>
          <h2>Actor: {result.actor || "Unknown"}</h2>

          {result.confidence !== undefined && (
            <p style={styles.confidence}>
              Confidence: {result.confidence.toFixed(2)}
            </p>
          )}

          {result.movies && result.movies.length > 0 ? (
            <>
              <h3>Movies/Shows:</h3>
              <ul style={styles.list}>
                {result.movies.map((movie, index) => (
                  <li key={index}>{movie}</li>
                ))}
              </ul>
            </>
          ) : (
            <p>No movies found or actor not recognized.</p>
          )}
        </div>
      )}
    </div>
  );
};

// Inline styles for better visual structure
const styles = {
  container: {
    textAlign: "center",
    padding: "30px",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  },
  title: {
    fontSize: "28px",
    marginBottom: "20px",
  },
  dropzone: {
    border: "2px dashed #aaa",
    padding: "25px",
    borderRadius: "12px",
    backgroundColor: "#f9f9f9",
    cursor: "pointer",
    marginBottom: "20px",
  },
  image: {
    marginTop: "20px",
    maxWidth: "300px",
    borderRadius: "10px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
  },
  resultBox: {
    marginTop: "30px",
    padding: "20px",
    borderRadius: "10px",
    backgroundColor: "#f0f0f0",
    display: "inline-block",
  },
  confidence: {
    fontSize: "16px",
    fontWeight: "bold",
    color: "#333",
    marginBottom: "10px",
  },
  list: {
    listStyleType: "none",
    padding: 0,
  },
  error: {
    color: "red",
    marginTop: "15px",
  },
};

export default App;
