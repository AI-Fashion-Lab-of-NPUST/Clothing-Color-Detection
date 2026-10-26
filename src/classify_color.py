import colorsys

r1, g1, b1 = 210, 205, 203
r2, g2, b2 = 176, 171, 168
r3, g3, b3 = 134, 126, 121

detected_clusters = [
    {"r": r1, "g": g1, "b": b1, "pct": 47.8},  # Main body
    {"r": r2, "g": g2, "b": b2, "pct": 31.7},  # Light shadow
    {"r": r3, "g": g3, "b": b3, "pct": 20.5}   # Deep shadow
]

def classify_hsv(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    
    h_deg = h * 360
    s_pct = s * 100
    v_pct = v * 100

    # CAMERA SHADOW CALIBRATION:
    # If a pixel has extremely low saturation, it means there is no actual color tint.
    # In real camera crops, a white shirt's shadows can easily drop to 40% brightness.
    if s_pct < 10: 
        if v_pct > 45:   # Lowered from 70 to 45 to absorb fabric fold shadows
            return "White"
        elif v_pct > 15: 
            return "Grey"
        else:            
            return "Black"

    # Rule B: Standard True Colors
    if (h_deg >= 0 and h_deg < 15) or (h_deg >= 330 and h_deg <= 360):
        if s_pct < 30 and v_pct > 70: return "Beige"
        elif v_pct > 70 and s_pct > 30: return "Pink" if s_pct < 60 else "Red"
        return "Red"
    elif h_deg >= 15 and h_deg < 45:
        return "Brown" if v_pct < 60 else "Beige"
    elif h_deg >= 45 and h_deg < 75:
        return "Yellow"
    elif h_deg >= 75 and h_deg < 165:
        return "Green"
    elif h_deg >= 165 and h_deg < 255:
        return "Blue"
    elif h_deg >= 255 and h_deg < 330:
        return "Purple"
    
    return "Grey"

final_predictions = {
    "White": 0.0, "Black": 0.0, "Grey": 0.0, "Red": 0.0, "Blue": 0.0,
    "Green": 0.0, "Yellow": 0.0, "Brown": 0.0, "Beige": 0.0, "Pink": 0.0, "Purple": 0.0
}

for cluster in detected_clusters:
    matched_label = classify_hsv(cluster["r"], cluster["g"], cluster["b"])
    final_predictions[matched_label] += cluster["pct"]

print("\n=== Calibrated AI Color Classification ===")
for color, percentage in final_predictions.items():
    if percentage > 5.0:
        print(f"👕 Predicted Color: {color} ({percentage:.1f}%)")
