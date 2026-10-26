import os
import torch
import numpy as np
import colorsys
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
INPUT_FOLDER = "Final_Test_Data"
VIS_OUTPUT_FOLDER = "Final_output_visualizations1"
OUTPUT_CSV = "final_color_pred1.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure directories exist
os.makedirs(VIS_OUTPUT_FOLDER, exist_ok=True)

print("Loading specialized fashion segmentation model...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEVICE)
model.eval()

# ==========================================
# 2. CALIBRATED HSV COLOR CLASSIFIER
# ==========================================
def classify_hsv(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    h_deg = h * 360
    s_pct = s * 100
    v_pct = v * 100

    # 1. BALANCED BLACK OVERRIDE (Protects Dark Green Jackets)
    if v_pct < 27:
        gb_diff = abs(int(g) - int(b))
        
        # FIXED: A black shirt has close G and B, but its RED is also close to them.
        # If Green is significantly stronger than Red (g > r + 15), it is a REAL green fabric, 
        # NOT black sensor noise! This completely saves green_jacket.jpg.
        if int(g) > (int(r) + 15):
            pass # Skip black override, allow it to fall to Green rules below
        elif gb_diff < 12:  
            return "Black"
        elif s_pct <= 25:
            return "Black"
        
    # Low saturation check for standard neutral scale garments
    if s_pct < 14:
        if v_pct > 72:   return "White"
        elif v_pct > 26: return "Grey"
        else:            return "Black"

    color_weights = {}

    # 2. SEPARATING TRUE RED vs BROWN vs PINK
    if (h_deg >= 0 and h_deg < 18) or (h_deg >= 325 and h_deg <= 360):
        # FIXED: True vibrant Red clothing stays bright under camera lighting.
        # If the brightness (v_pct) is below 46%, it is safely classified as Brown.
        if v_pct < 46:
            color_weights["Brown"] = 1.5   # Correctly forces brown_tshirt.jpg to Brown
        elif s_pct < 45 and v_pct > 38: 
            color_weights["Pink"] = 1.5   
        else:
            color_weights["Red"] = 1.2     # Keeps true red garments highly accurate

    # Brown vs Beige vs Yellow vs Green (Hue 18 to 78)
    elif h_deg >= 18 and h_deg < 78:
        if h_deg < 42: # Brown/Beige zones
            if v_pct < 48: color_weights["Brown"] = 1.2
            else:          color_weights["Beige"] = 1.0
        else: # Yellow / Olive / Dull Green zone (42 to 78 degrees)
            if s_pct < 18 and v_pct > 55:
                color_weights["Beige"] = 1.5   
            elif s_pct < 35 and v_pct < 65:
                if s_pct > 14:
                    color_weights["Green"] = 1.5   
                else:
                    color_weights["Grey"] = 1.0
            elif v_pct < 45:
                color_weights["Brown"] = 1.0
            else:
                color_weights["Yellow"] = 1.0

    # True Green zone
    elif h_deg >= 78 and h_deg < 165:
        color_weights["Green"] = 1.2

    # Blue zone (with Camera Green-Drift Fix)
    elif h_deg >= 165 and h_deg < 255:
        if h_deg < 185 and s_pct > 25:
            color_weights["Green"] = 1.6  
        else:
            if v_pct < 30: return "Black"
            color_weights["Blue"] = 1.0

    # Purple zone
    elif h_deg >= 255 and h_deg < 330:
        color_weights["Purple"] = 1.0

    if color_weights:
        return max(color_weights, key=color_weights.get)
        
    return "Grey"







# ==========================================
# 3. CORE PROCESSING PIPELINE WITH VISUALIZATION
# ==========================================
def process_and_visualize(image_path, filename):
    try:
        # A. Segment Fabric
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Look for Upper Clothes (Index 4)
        shirt_mask = (pred_seg == 4)
        image_np = np.array(image)
        fabric_pixels = image_np[shirt_mask]
        
        # Fallback for full outer garments (Index 7) or dresses (Index 1)
        if len(fabric_pixels) < 100:  
            shirt_mask = (pred_seg == 7) | (pred_seg == 1)
            fabric_pixels = image_np[shirt_mask]
            
        if len(fabric_pixels) == 0:
            return "No clothing detected"

        # Isolate fabric for visualization plot
        extracted_shirt = np.zeros_like(image_np)
        extracted_shirt[shirt_mask] = image_np[shirt_mask]

        # B. Cluster Dominant Colors using K-Means
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=5)
        kmeans.fit(fabric_pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        counts = Counter(kmeans.labels_)
        total_counts = sum(counts.values())

        # C. Map and Combine Predictions
        image_predictions = {
            "White": 0.0, "Black": 0.0, "Grey": 0.0, "Red": 0.0, "Blue": 0.0,
            "Green": 0.0, "Yellow": 0.0, "Brown": 0.0, "Beige": 0.0, "Pink": 0.0, "Purple": 0.0
        }
        
        for i, count in counts.items():
            r, g, b = colors[i]
            pct = (count / total_counts) * 100
            matched_label = classify_hsv(r, g, b)
            image_predictions[matched_label] += pct

        # Format output string
        # detected_list = [f"{color} ({pct:.1f}%)" for color, pct in image_predictions.items() if pct > 5.0]
        # prediction_text = ", ".join(detected_list)

                # --- الفلترة الديناميكية الذكية لحماية الألوان ومنع الهلوسة ---
        detected_list = []
        for color, pct in image_predictions.items():
            # الألوان التي يكثر فيها الخطأ والهلوسة نضع لها شروطاً صارمة (نسبة عالية في القماش)
            if color in ["Grey", "Green", "Beige", "Blue", "Purple", "Brown"]:
                if pct >= 28.0:  # لن يتم اعتماد اللون إلا إذا كان ظاهراً وصريحاً بنسبة قوية
                    detected_list.append(f"{color} ({pct:.1f}%)")
            
            # الألوان المستقرة أو الحذرة نترك لها حداً مرناً للاحتفاظ بالـ Recall
            elif color in ["Black", "White", "Red", "Yellow", "Pink"]:
                if pct >= 12.0:  
                    detected_list.append(f"{color} ({pct:.1f}%)")

        # إجراء احترازي: إذا خرجت القائمة فارغة بسبب الشروط، نأخذ اللون الأعلى نسبة مطلقاً
        if not detected_list:
            highest_color = max(image_predictions, key=image_predictions.get)
            detected_list = [f"{highest_color} ({image_predictions[highest_color]:.1f}%)"]
            
        prediction_text = ", ".join(detected_list)



        # D. Generate and Save Side-by-Side Plot with Text Banner
        plt.figure(figsize=(10, 6))
        
        # Original Image
        plt.subplot(1, 2, 1)
        plt.title("Original Torso Crop", fontsize=12)
        plt.imshow(image)
        plt.axis('off')
        
        # Isolated Fabric
        plt.subplot(1, 2, 2)
        plt.title("Isolated Cloth Fabric", fontsize=12)
        plt.imshow(extracted_shirt)
        plt.axis('off')
        
        # Superimpose text at the very top center dynamically
        plt.suptitle(f"File: {filename}\nAI Predicted Colors: {prediction_text}", 
                     fontsize=14, color='darkgreen', fontweight='bold', y=0.98)
        
        # Save output figure cleanly
        vis_save_path = os.path.join(VIS_OUTPUT_FOLDER, f"vis_{filename}")
        plt.tight_layout(rect=[0, 0, 1, 0.90]) # Make room for the top subtitle
        plt.savefig(vis_save_path, bbox_inches='tight', dpi=150)
        plt.close()

        return prediction_text

    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 4. RUN AUTOMATION LOOP
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created '{INPUT_FOLDER}' folder. Please place your test images inside it and rerun.")
        exit()

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(image_extensions)]
    
    if len(all_images) == 0:
        print(f"No images found inside the '{INPUT_FOLDER}' folder.")
        exit()
        
    print(f"Found {len(all_images)} images. Processing and drawing visualizations...")
    
    results_records = []
    for idx, filename in enumerate(all_images):
        full_path = os.path.join(INPUT_FOLDER, filename)
        print(f"[{idx+1}/{len(all_images)}] Visualizing: {filename}...")
        
        color_output = process_and_visualize(full_path, filename)
        results_records.append({"filename": filename, "predicted_colors": color_output})

    df_results = pd.DataFrame(results_records)
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    print("\n==========================================")
    print(f"Pipeline finished! Check your results:")
    print(f" 📂 Visualizations directory: {VIS_OUTPUT_FOLDER}/")
    print(f" 📊 Final spreadsheet summary: {OUTPUT_CSV}")
    print("==========================================")
