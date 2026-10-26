from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Your low-quality real camera torso crop path
IMAGE_PATH = r"C:\Users\admin\Desktop\Outfit-Color-Understanding-with-Segmentation-Calibration-XKCD-Naming-main\test_data\white_tshirt_top.jpg"

# 2. Automatically load the dedicated fashion parsing model from Hugging Face
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# 3. Read and pre-process the image
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# 4. Predict the masks
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 5. Resize back to original image dimensions
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1).squeeze().numpy()

# 6. Plot and save the results side-by-side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Torso Crop")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Detected Fashion Mask")
plt.imshow(pred_seg, cmap="tab20")

# Save visualization
plt.savefig("fashion_seg_result.png")
print("Done! Please open 'fashion_seg_result.png' to check the mask accuracy.")
