import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def get_dominant_colors(image_path, num_colors=3):
    # 1. Load the isolated fabric image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Reshape the image to a list of pixels
    pixels = image.reshape(-1, 3)
    
    # 3. CRITICAL: Filter out the pure black background pixels (0, 0, 0)
    # We only keep pixels where at least one channel (R, G, B) is greater than 10
    fabric_pixels = pixels[np.any(pixels > 10, axis=1)]
    
    if len(fabric_pixels) == 0:
        return "No fabric detected"
    
    # 4. Use K-Means to cluster the remaining clothing colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(fabric_pixels)
    
    # 5. Get the cluster center colors and their percentages
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    counts = Counter(labels)
    total_counts = sum(counts.values())
    
    # 6. Print the detected RGB values and their distribution percentages
    print("\n--- Detected Fabric Colors ---")
    for i, count in counts.most_common():
        color_rgb = colors[i]
        percentage = (count / total_counts) * 100
        print(f"Color RGB: {list(color_rgb)} -> Percentage: {percentage:.1f}%")

# Run the color extractor on your isolated shirt image
get_dominant_colors("only_shirt_fabric.png", num_colors=3)
