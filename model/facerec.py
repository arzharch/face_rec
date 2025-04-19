import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import insightface

# Initialize face detection and recognition model using SCRFD + ArcFace on CPU
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640, 640))

# Extract 512-dimensional face embedding from image
def get_embedding(img):
    faces = model.get(img)
    if len(faces) == 0:
        return None  # No face found
    return faces[0].embedding  # Return first detected face's embedding

# Load dataset images and compute face embeddings
def load_dataset(dataset_path):
    X, y = [], []
    for person in os.listdir(dataset_path):  # Loop through each person's folder
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path): 
            continue
        
        print(f"Processing images for: {person}")  # Print the name of the person being processed
        
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path)
            if img is None: 
                print(f"Skipping invalid image: {img_path}")  # Log invalid images
                continue  # Skip if image is not loaded properly
            
            embedding = get_embedding(img)
            if embedding is not None:
                X.append(embedding)
                y.append(person)  # Label as folder name
            else:
                print(f"No face detected in image: {img_path}")  # Log images with no detected face
    return np.array(X), np.array(y)

# Load data from specified dataset path
dataset_path = r'C:\Users\arshc\Desktop\face_rec\Celebrity Faces Dataset'
X, y = load_dataset(dataset_path)

# Encode string labels into numerical format
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM classifier using linear kernel
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Predict on test set and calculate accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Validation Accuracy: {acc:.2f}")

# Save trained classifier and label encoder to disk
joblib.dump(clf, "svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("💾 Model and label encoder saved.")