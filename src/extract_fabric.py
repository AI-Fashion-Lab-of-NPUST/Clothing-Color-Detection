from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch
import numpy as np
from PIL import Image

IMAGE_PATH = r"C:\Users\admin\Desktop\Outfit-Color-Understanding-with-Segmentation-Calibration-XKCD-Naming-main\test_data\white_tshirt_top.jpg"

# 1. Load the model and process image
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 2. Resize mask to match original image dimensions
upsampled_logits = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
pred_seg = upsampled_logits.argmax(dim=1).squeeze().numpy()

# 3. Create a clean mask for Upper Clothes (Index 4)
# This creates an array of True where the shirt is, and False everywhere else
shirt_mask = (pred_seg == 4)

# 4. Extract only the shirt pixels from the original image
image_np = np.array(image)
extracted_shirt = np.zeros_like(image_np) # Start with a completely black image

# Copy over ONLY the pixels where our shirt mask is True
extracted_shirt[shirt_mask] = image_np[shirt_mask]

# 5. Save the isolated fabric image
output_image = Image.fromarray(extracted_shirt)
output_image.save("only_shirt_fabric.png")

print("Success! 'only_shirt_fabric.png' has been saved.")
