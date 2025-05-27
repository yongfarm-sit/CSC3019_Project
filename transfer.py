import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# === CONFIG ===
DATA_DIR = './dataset_2' #training dataset directory
TEST_DIR = './test'  # test dataset directory
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 10


# === LOAD DATASETS ===
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

# === NORMALIZATION ===
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(normalize).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(normalize).cache().prefetch(buffer_size=AUTOTUNE)

# === VISUALIZATION ===
def plot_images(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    
# === MODEL ===
def transfer_model():
    base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def compile_and_train(model):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    return history

def evaluate_model(model, dataset, name="Validation"):
    loss, acc = model.evaluate(dataset)
    print(f"{name} Loss: {loss:.4f}, {name} Accuracy: {acc:.4f}")

    y_true = []
    y_pred = []
    
    for images, labels in dataset:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print(f"\n{name} Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# === MAIN ===
if __name__ == "__main__":
    plot_images(train_ds)

    print("\nBuilding the model...")
    model = transfer_model()

    print("\nTraining the model...")
    compile_and_train(model)

    print("\nEvaluating on validation set...")
    evaluate_model(model, val_ds, name="Validation")

    print("\nEvaluating on test set...")
    evaluate_model(model, test_ds, name="Test")

    print("\nSaving model...")
    model.save('transfer_model.h5')

    print("\n✅ Model training and test evaluation completed.")
