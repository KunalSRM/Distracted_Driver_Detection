import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools

# Paths
dataset_dir = 'dataset/imgs'
history_path = 'history.npy'
model_path = 'model/distracted_driver_model.h5'

# Load training history
history = np.load(history_path, allow_pickle=True).item()

# ==== 1. Plot Training vs Validation Accuracy ====
plt.figure(figsize=(8, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
# I tell you how the model’s accuracy improved for training and validation over time
plt.savefig("graph_accuracy.png")
plt.close()

# ==== 2. Plot Training vs Validation Loss ====
plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# I tell you how the model’s loss reduced for both training and validation as epochs progressed
plt.savefig("graph_loss.png")
plt.close()

# ==== 3. Class Distribution ====
train_dir = os.path.join(dataset_dir, 'train')
classes = os.listdir(train_dir)
class_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in classes]

plt.figure(figsize=(10, 6))
sns.barplot(x=classes, y=class_counts)
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
# I tell you how many images are present in each class in the dataset
plt.savefig("graph_class_distribution.png")
plt.close()

# ==== 4. Sample Images from Each Class ====
plt.figure(figsize=(15, 10))
for idx, cls in enumerate(classes):
    cls_dir = os.path.join(train_dir, cls)
    img_path = os.listdir(cls_dir)[0]  # first image
    img = plt.imread(os.path.join(cls_dir, img_path))
    plt.subplot(3, 4, idx+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis('off')
# I tell you what sample images from each driving class look like
plt.tight_layout()
plt.savefig("graph_sample_images.png")
plt.close()

# ==== Load model for predictions ====
model = load_model(model_path)

# Prepare test data generator
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predictions
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ==== 5. Confusion Matrix ====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
# I tell you how well the model predicted each class, showing correct and incorrect predictions
plt.savefig("graph_confusion_matrix.png")
plt.close()

# ==== 6. Per-Class Accuracy ====
per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
sns.barplot(x=class_labels, y=per_class_acc)
plt.ylim(0, 1)
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
# I tell you the accuracy for each class separately, so you know which classes the model is strong or weak at
plt.savefig("graph_per_class_accuracy.png")
plt.close()

# ==== 7. ROC Curves (One vs All) ====
plt.figure(figsize=(12, 8))
for i in range(len(class_labels)):
    fpr, tpr, _ = roc_curve(y_true == i, preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves (One vs All)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
# I tell you how well the model separates each class from all others, using ROC curves and AUC values
plt.savefig("graph_roc_curves.png")
plt.close()

print("✅ All graphs have been generated and saved successfully!")
