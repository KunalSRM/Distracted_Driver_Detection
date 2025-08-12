import pandas as pd

# Load the CSV that has all image names + class names
df = pd.read_csv('dataset/driver_imgs_list.csv')

# Map c0 → 0, c1 → 1, ... c9 → 9
class_map = {f'c{i}': i for i in range(10)}
df['label'] = df['classname'].map(class_map)

# Keep only filename and label
df_labels = df[['img', 'label']].rename(columns={'img': 'filename'})

# Save as test_labels.csv in project root
df_labels.to_csv('test_labels.csv', index=False)

print("✅ test_labels.csv created successfully!")
