import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = load_model('model/distracted_driver_model.h5')

# Paths
test_folder = 'dataset/imgs/test'  # All test images in this folder
labels_file = 'test_labels.csv'    # filename,label

# Load CSV
df = pd.read_csv(labels_file)

# Filter only files that actually exist
df = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(test_folder, x)))]
print(f"✅ Found {len(df)} valid images in '{test_folder}'")

# Prepare data
X = []
y_true = []

for _, row in df.iterrows():
    img_path = os.path.join(test_folder, row['filename'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (224, 224)) / 255.0
    X.append(img)
    y_true.append(row['label'])

X = np.array(X)
y_true = np.array(y_true)

if len(X) == 0:
    raise ValueError("❌ No images found! Check your test folder and CSV filenames.")

# Predict
predictions = model.predict(X, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification report
print(classification_report(y_true, y_pred))
