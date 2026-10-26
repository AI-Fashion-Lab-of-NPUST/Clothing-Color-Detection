import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

# 1. LOAD DATASETS
pred_df = pd.read_csv("final_color_pred1.csv")
true_df = pd.read_csv("color_labels.csv")

# Combine files by matching filenames
merged = pd.merge(pred_df, true_df, on="filename")

presence_correct = 0
total_valid_images = 0

# Lists for Rule 2 (Color Presence - Multi Label)
y_true_multi = []
y_pred_multi = []

# Standard 11 classes to lock our evaluation shape
VALID_LABELS = ['Beige', 'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Pink', 'Purple', 'Red', 'White', 'Yellow']

# Create output folder for results
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# 2. PROCESS DATA
for idx, row in merged.iterrows():
    pred_string = str(row['predicted_colors'])
    true_color = str(row['label']).strip()
    
    # Skip errors or unlabelled targets
    if "No clothing detected" in pred_string or pd.isna(row['predicted_colors']) or true_color.lower() == "unknown":
        continue
        
    total_valid_images += 1
    
    # Clean up predictions: "White (73.9%), Grey (26.1%)" -> ['White', 'Grey']
    detected_colors = [item.split("(")[0].strip() for item in pred_string.split(",")]
    
    # --- Append Data for Rule 2 (Presence) ---
    y_true_multi.append([true_color])
    y_pred_multi.append(detected_colors)
    
    if true_color.lower() in [c.lower() for c in detected_colors]:
        presence_correct += 1

# Calculate Accuracy
presence_acc = (presence_correct / total_valid_images) * 100 if total_valid_images > 0 else 0

# 3. MULTI-LABEL BINARIZATION
mlb = MultiLabelBinarizer(classes=VALID_LABELS)
y_true_bin = mlb.fit_transform(y_true_multi)
y_pred_bin = mlb.transform(y_pred_multi)

# 4. GENERATE REPORTS & TEXT OUTPUTS
report_dict = classification_report(y_true_bin, y_pred_bin, target_names=VALID_LABELS, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{output_dir}/classification_report.csv", index=True)

summary_text = (
    f"=======================================================\n"
    f"📊 SYSTEM-WIDE ACCURACY SUMMARY ({total_valid_images} Total Samples)\n"
    f"=======================================================\n"
    f"🎨 Rule 2: Any-Color Presence Accuracy:   {presence_acc:.2f}%\n"
    f"=======================================================\n\n"
    f"📝 MULTI-LABEL CLASSIFICATION REPORT\n"
    f"-------------------------------------------------------\n"
    f"{classification_report(y_true_bin, y_pred_bin, target_names=VALID_LABELS, zero_division=0)}"
)

# Print to console and save to text file
print(summary_text)
with open(f"{output_dir}/summary_report.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

# 5. VISUALIZATION 1: Multi-Label Confusion Matrices (FIXED AXIS LAYOUT)
mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for i, label in enumerate(VALID_LABELS):
    # We slice and rearrange the matrix to force standard layout:
    # Top Row: [True Presence], Bottom Row: [True Absence]
    # Left Column: [Pred Presence], Right Column: [Pred Absence]
    tn, fp, fn, tp = mcm[i].ravel()
    fixed_cm = np.array([[tp, fn], 
                         [fp, tn]])
    
    sns.heatmap(fixed_cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=['Pred Pres', 'Pred Abs'], 
                yticklabels=['True Pres', 'True Abs'])
    axes[i].set_title(f'Color: {label}')

if len(VALID_LABELS) < len(axes):
    axes[-1].axis('off')

plt.suptitle("Rule 2: Corrected Confusion Matrix Per Color Class", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/multi_label_confusion_matrices.png", dpi=300)
plt.close()


# 6. VISUALIZATION 2: Co-occurrence Matrix (What pairs get predicted together)
co_occurrence = np.dot(y_pred_bin.T, y_pred_bin)
np.fill_diagonal(co_occurrence, 0) # Fill diagonal with 0 to focus on overlap pairings
co_occurrence_df = pd.DataFrame(co_occurrence, index=VALID_LABELS, columns=VALID_LABELS)

plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence_df, annot=True, fmt='d', cmap='Purples')
plt.title("🎨 Color Co-occurrence in Predictions (Rule 2 Overlaps)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/color_co_occurrence.png", dpi=300)
plt.close()

print(f"💾 All evaluation targets successfully saved to the folder: '{output_dir}/'")
