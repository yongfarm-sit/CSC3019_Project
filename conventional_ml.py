import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Paths and constants
DATA_DIR = './dataset_2'
TEST_DIR = './test'
IMG_SIZE = (128, 128)

# === Feature Extraction for Conventional ML ===
def extract_features_and_labels(data_dir):
    features, labels = [], []
    base_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(128, 128, 3))

    class_names = sorted(os.listdir(data_dir))
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            feature = base_model.predict(img_array, verbose=0)
            features.append(feature.squeeze())
            labels.append(label_idx)
    
    return np.array(features), np.array(labels), class_names

# === Extract test features ===
def extract_test_features(test_dir, class_names):
    features, labels = [], []
    base_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(128, 128, 3))

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            feature = base_model.predict(img_array, verbose=0)
            features.append(feature.squeeze())
            labels.append(label_idx)

    return np.array(features), np.array(labels)

# === Train and Evaluate SVM ===
def train_ml_classifier(X, y, class_names):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("\nConventional ML Validation Set Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    print("Conventional ML Validation Set Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
    return clf, scaler

    
def evaluate_on_test_set(clf, scaler, X_test, y_test, class_names):
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    print("\nConventional ML Test Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Conventional ML Test Set Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# === Main Execution ===
if __name__ == "__main__":
    print("\n🔍 Extracting features using MobileNetV2...")
    X, y, class_names = extract_features_and_labels(DATA_DIR)

    print("\n🧠 Training SVM using extracted features...")
    clf, scaler = train_ml_classifier(X, y, class_names)

    print("\n🔍 Extracting test features...")
    X_test, y_test = extract_test_features(TEST_DIR, class_names)

    print("\n📊 Evaluating SVM on test dataset...")
    evaluate_on_test_set(clf, scaler, X_test, y_test, class_names)

    print("\n✅ Conventional ML pipeline completed.")
