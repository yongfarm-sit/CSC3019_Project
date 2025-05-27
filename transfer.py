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

DATA_DIR = './dataset_2'  # Update this path to your dataset directory
IMG_SIZE = (128,128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 10


# Load the dataset
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

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# Normalize the pixel values
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(normalize, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

# Prefetch

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Visualize some training images
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
    
plot_images(train_ds)

# Build the model

def transfer_model():
    base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


# Train the model
def compile_and_train(model):
    model.compile(
        optimizer='adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    return history

def evaluate_model(model):
    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Generate predictions
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        logits = model.predict(images)
        prediction = np.argmax(logits, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(prediction)

    print("Classification report: ")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    
    
# Main function to run the training and evaluation
if __name__ == "__main__":
    plot_images(train_ds)

    print("\nBuilding the model...")
    model = transfer_model()

    print("\nTraining the model...")
    compile_and_train(model)

    print("\nEvaluating the model...")
    evaluate_model(model)

    print("\nSaving model...")
    model.save('transfer_model.h5')

    print("\n✅ Model training and evaluation completed.")
