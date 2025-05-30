import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
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

# === Train one-vs-rest binary SVM classifiers with hyperparameter tuning ===
def train_binary_classifiers_with_tuning(X, y, class_names):
    classifiers = {}
    scalers = {}

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    for idx, class_name in enumerate(class_names):
        print(f"\n🧠 Training binary SVM for class: '{class_name}' (One-vs-Rest) with hyperparameter tuning")
        y_binary = (y == idx).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        svc = SVC(probability=True)
        grid = GridSearchCV(svc, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_clf = grid.best_estimator_

        y_pred = best_clf.predict(X_val_scaled)

        print(f"Best params for class '{class_name}': {grid.best_params_}")
        print(f"\n📋 Classification Report for class '{class_name}':")
        print(classification_report(y_val, y_pred, target_names=['Not '+class_name, class_name]))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

        classifiers[class_name] = best_clf
        scalers[class_name] = scaler

    return classifiers, scalers

# === Evaluate on test set using all binary classifiers combined ===
def evaluate_one_vs_rest(classifiers, scalers, X_test, y_test, class_names):
    print("\n📊 Evaluating one-vs-rest classifiers on test dataset...")
    n_samples = X_test.shape[0]
    n_classes = len(class_names)
    probs = np.zeros((n_samples, n_classes))

    for idx, class_name in enumerate(class_names):
        scaler = scalers[class_name]
        clf = classifiers[class_name]
        X_test_scaled = scaler.transform(X_test)
        probs[:, idx] = clf.predict_proba(X_test_scaled)[:, 1]

    # Final prediction = class with highest probability
    y_pred = np.argmax(probs, axis=1)

    print("\n🧠 Final Multi-class Prediction Report (from binary classifiers):")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Plot ROC Curves ===
def plot_roc_curves(classifiers, scalers, X_test, y_test, class_names):
    plt.figure(figsize=(10, 8))

    for idx, class_name in enumerate(class_names):
        scaler = scalers[class_name]
        clf = classifiers[class_name]

        X_test_scaled = scaler.transform(X_test)
        y_true = (y_test == idx).astype(int)
        y_scores = clf.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for One-vs-Rest SVM classifiers")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# === Perform EDA (same as your original function) ===
def perform_eda(data_dir):
    import seaborn as sns
    from collections import defaultdict

    print("\n🔎 Performing Exploratory Data Analysis (EDA)...")
    image_count = 0
    class_image_counts = {}
    image_shapes = []
    corrupted_images = []
    sample_images = defaultdict(list)

    class_names = sorted(os.listdir(data_dir))
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        image_files = os.listdir(class_path)
        class_image_counts[class_name] = len(image_files)
        image_count += len(image_files)
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(class_path, img_name)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path)
                image_shapes.append(img.size)
                if i < 3:  # Collect a few sample images for each class
                    sample_images[class_name].append(img)
            except Exception as e:
                print(f"⚠️ Error loading image {img_path}: {e}")
                corrupted_images.append(img_path)

    # Dataset overview
    print(f"\n📊 Total Images: {image_count}")
    print("📂 Images per class:")
    for class_name, count in class_image_counts.items():
        print(f"  {class_name}: {count}")
    
    # Class distribution bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(class_image_counts.keys()), y=list(class_image_counts.values()))
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

    # Image dimension variability
    unique_shapes = set(image_shapes)
    print(f"\n🖼️ Unique Image Dimensions Found: {len(unique_shapes)}")
    for shape in unique_shapes:
        print(f"  {shape}")

    # Show sample images
    for class_name in class_names:
        imgs = sample_images[class_name]
        if imgs:
            plt.figure(figsize=(10, 2))
            for i, img in enumerate(imgs):
                plt.subplot(1, 3, i + 1)
                plt.imshow(img)
                plt.axis("off")
                plt.title(class_name)
            plt.suptitle(f"Samples from class: {class_name}")
            plt.tight_layout()
            plt.show()

    # Report corrupted images
    if corrupted_images:
        print(f"\n❌ Found {len(corrupted_images)} corrupted or unreadable images.")
    else:
        print("\n✅ No corrupted images found.")

# === Main Execution ===
if __name__ == "__main__":
    print("\n🔍 Starting Exploratory Data Analysis...")
    perform_eda(DATA_DIR)

    print("\n🔍 Extracting features using MobileNetV2...")
    X, y, class_names = extract_features_and_labels(DATA_DIR)

    print("\n🧠 Training one-vs-rest SVM classifiers with tuning...")
    classifiers, scalers = train_binary_classifiers_with_tuning(X, y, class_names)

    print("\n🔍 Extracting test features...")
    X_test, y_test = extract_test_features(TEST_DIR, class_names)

    print("\n📊 Evaluating one-vs-rest classifiers on test dataset...")
    evaluate_one_vs_rest(classifiers, scalers, X_test, y_test, class_names)

    print("\n📈 Plotting ROC Curves...")
    plot_roc_curves(classifiers, scalers, X_test, y_test, class_names)

    print("\n✅ Conventional ML pipeline with fine-tuning completed.")

