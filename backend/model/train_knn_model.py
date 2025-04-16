# === Imports ===
import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import faiss

# === CONFIGURATION ===
# Set image size for MobileNetV2 input
IMG_SIZE = 96

# Paths to datasets and model output
safe_path = r"C:\Users\shree\OneDrive\Desktop\projects\botrush\hackactivate_botrush_underfitted_legends\dataset\Safe"
unsafe_path = r"C:\Users\shree\OneDrive\Desktop\projects\botrush\hackactivate_botrush_underfitted_legends\dataset\Unsafe"
model_output_path = r"C:\Users\shree\OneDrive\Desktop\projects\botrush\hackactivate_botrush_underfitted_legends\models\knn_model.pkl"

# === Feature Extractor Setup ===
# Load MobileNetV2 model pre-trained on ImageNet, remove the top layer, use global average pooling
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
# Use this modified model as the feature extractor
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# === Function to Extract Features from a Single Image ===
def extract_features_from_image(img):
    # Resize image to match model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Preprocess for MobileNetV2
    img = preprocess_input(img.astype(np.float32))
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    # Extract features using MobileNetV2
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()

# === Function to Extract Features from All Images in a Folder ===
def extract_features_from_folder(folder_path, label):
    features, labels = [], []
    # Walk through all files in folder (and subfolders)
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, fname)
                img = cv2.imread(path)
                if img is None:
                    print(f"Skipping unreadable file: {path}")
                    continue
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Extract features and save with label
                feat = extract_features_from_image(img)
                features.append(feat)
                labels.append(label)
    return features, labels

# === Load and Extract Features from Both Safe and Unsafe Datasets ===
print("Extracting safe features...")
X_safe, y_safe = extract_features_from_folder(safe_path, 0)  # Label 0 for safe
print("Extracting unsafe features...")
X_unsafe, y_unsafe = extract_features_from_folder(unsafe_path, 1)  # Label 1 for unsafe

# Combine all features and labels
X = np.vstack(X_safe + X_unsafe).astype('float32')
y = np.array(y_safe + y_unsafe)

print("Total features extracted:", X.shape[0])

# === Split Data into Training and Testing Sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train K-Nearest Neighbors Classifier ===
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# === Evaluate Model Performance on Test Set ===
y_pred = knn.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# === Save Trained KNN Model ===
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
joblib.dump(knn, model_output_path)
print(f"Model saved to {model_output_path}")

# === Save Features and Labels for Later Use (Optional) ===
np.save("features.npy", X)
np.save("labels.npy", y)

# === Create and Save FAISS Index for Fast Similarity Search ===
def create_faiss_index(features):
    dim = features.shape[1]  # Dimensionality of feature vectors
    index = faiss.IndexFlatL2(dim)  # Use L2 distance metric
    index.add(features)  # Add feature vectors to index
    return index

# Create FAISS index from feature matrix
faiss_index = create_faiss_index(X)

# Save FAISS index to disk
faiss.write_index(faiss_index, "faiss_index.index")
