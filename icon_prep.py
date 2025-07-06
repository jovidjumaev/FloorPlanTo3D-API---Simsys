# icon_crop_alpha.py
import sys
import cv2
import numpy as np
from pathlib import Path

def crop_to_alpha(in_file: str, out_file: str):
    # 1) load with alpha
    img = cv2.imread(in_file, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"couldn't read {in_file}")
    if img.shape[2] < 4:
        # no alpha channel, fall back to Otsu on gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        b, g, r = cv2.split(img)
        img = cv2.merge([b, g, r, alpha])
    else:
        # already 4-channel
        b, g, r, alpha = cv2.split(img)

    # 2) find non-zero alpha coords
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        # nothing to crop
        crop = img
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        # optional 1px pad:
        y0, x0 = max(0, y0-1), max(0, x0-1)
        y1 = min(img.shape[0]-1, y1+1)
        x1 = min(img.shape[1]-1, x1+1)
        crop = img[y0:y1+1, x0:x1+1]

    # 3) write out
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_file, crop)
    print(f"→ cropped to alpha bbox and saved {out_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python icon_crop_alpha.py input.png output.png")
        sys.exit(1)
    crop_to_alpha(sys.argv[1], sys.argv[2])
