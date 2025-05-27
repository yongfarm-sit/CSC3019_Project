import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Paths and constants
DATA_DIR = './dataset_2'
TEST_DIR = './test'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 10

# Load datasets
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)
val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)
test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# Normalize datasets
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(normalize).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(normalize).cache().prefetch(buffer_size=AUTOTUNE)

# Build simple CNN model
def simple_cnn():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Compile and train model
def compile_and_train(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    return history

# Evaluate model and print metrics
def evaluate_model(model, dataset, name="Dataset"):
    loss, accuracy = model.evaluate(dataset)
    print(f"\n{name} Loss: {loss:.4f}, {name} Accuracy: {accuracy:.4f}")
    
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

# Main
if __name__ == "__main__":
    print("\nBuilding and training CNN...")
    model = simple_cnn()
    compile_and_train(model)
    
    print("\nEvaluating CNN on validation set...")
    evaluate_model(model, val_ds, name="Validation")
    
    print("\nEvaluating CNN on test set...")
    evaluate_model(model, test_ds, name="Test")
    
    print("\nSaving CNN model...")
    model.save('simple_cnn_model.h5')
