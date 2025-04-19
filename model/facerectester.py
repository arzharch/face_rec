import cv2
import joblib
import insightface
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Load pre-trained SVM classifier and label encoder
clf = joblib.load("svm_model.pkl")
le = joblib.load("label_encoder.pkl")

# Initialize insightface model for face detection and embedding
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640, 640))

# Launch GUI to browse and recognize face from image
def browse_and_recognize():
    # Hide root window of Tkinter
    root = Tk()
    root.withdraw()

    # Open file dialog to select image
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("No file selected.")
        return

    # Load selected image
    img = cv2.imread(file_path)
    if img is None:
        print("Failed to read image.")
        return

    # Detect faces and get embedding
    faces = model.get(img)
    if not faces:
        print("No face detected.")
        return

    face = faces[0]  # Use the first detected face
    embedding = face.embedding

    # Predict identity using the trained SVM model
    pred = clf.predict([embedding])[0]
    label = le.inverse_transform([pred])[0]

    # Draw bounding box and predicted label on the image
    (x, y, w, h) = face.bbox.astype(int)
    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Convert image to RGB and show with prediction
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {label}")
    plt.axis('off')
    plt.show()

# Start recognition workflow
browse_and_recognize()