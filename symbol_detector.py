import cv2, numpy as np, math
from pathlib import Path
import time
import unicodedata

# Helper to rotate without cropping (placed BEFORE template loading)
import cv2
import numpy as np

def _rotate_image_bound(img, angle):
    """Rotate image without cropping (returns a new image)"""
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=255)

# -------------------------
# Load templates (only camera / sensor icons)
# -------------------------
ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"  # single folder with all PNGs
TEMPLATES = {}

# ------------------
# Debug flag
# ------------------
DEBUG_LOAD = True  # set True to print why templates are kept / discarded

def _categorize_icon(stem: str):
    stem_norm = unicodedata.normalize('NFC', stem)
    lower = stem_norm.lower()
    # Simple heuristics: include both English and Korean keywords
    if "camera" in lower or "카메라" in stem_norm:
        return "camera"
    if "sensor" in lower or "감지기" in stem_norm:
        return "sensor"
    return None

CLASS_MAP = {}
DISPLAY_NAME = {}

for p in ICON_DIR.glob("*.png"):
    stem = p.stem
    category = _categorize_icon(stem)
    if category is None:
        continue  # ignore other templates

    template_raw = cv2.imread(str(p), 0)
    if template_raw is None or template_raw.size == 0:
        continue

    # Skip big or complex drawings (likely product photos)
    h_raw, w_raw = template_raw.shape[:2]
    if max(h_raw, w_raw) > 120:
        if DEBUG_LOAD:
            print(f"Skip {p.name}: too large ({w_raw}x{h_raw})")
        continue

    # Assign class ID based on category
    cid = 4 if category == "camera" else 5  # 4:Camera, 5:Sensor

    # Store grayscale raw template (better keypoints at small sizes)
    TEMPLATES[stem] = template_raw
    CLASS_MAP[stem] = cid
    DISPLAY_NAME[stem] = category.capitalize()

    # Add rotated versions every 45° (excluding 0° which is already added)
    for angle in range(45, 360, 45):
        rotated_raw = _rotate_image_bound(template_raw, angle)
        r_name = f"{stem}_rot{angle}"
        TEMPLATES[r_name] = rotated_raw
        CLASS_MAP[r_name] = cid
        DISPLAY_NAME[r_name] = DISPLAY_NAME[stem]

    # Add down-scaled copies (0.75, 0.6, 0.5, 0.3) to cover small glyphs
    for scale in (0.75, 0.6, 0.5, 0.3):
        if min(h_raw, w_raw) * scale < 20:  # avoid making it too tiny
            continue
        scaled_raw = cv2.resize(template_raw, (int(w_raw*scale), int(h_raw*scale)), interpolation=cv2.INTER_AREA)
        s_name = f"{stem}_s{int(scale*100)}"
        TEMPLATES[s_name] = scaled_raw
        CLASS_MAP[s_name] = cid
        DISPLAY_NAME[s_name] = DISPLAY_NAME[stem]

    if DEBUG_LOAD:
        print(f"Keep {p.name}: size {w_raw}x{h_raw}, edge density {np.count_nonzero(template_raw)/(h_raw*w_raw):.2f}")

# Precompute template areas for quick size checks
TEMPLATE_SIZES = {name: tmpl.shape[:2] for name, tmpl in TEMPLATES.items()}  # (h, w)

# Scale variants to try per template (relative to detected global scale)
SCALE_VARIANTS = [0.8, 1.0, 1.25]

def _resize_image_keep_ratio(img, max_dim):
    """Resize so that the longest dimension == max_dim, keep aspect."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / float(max(h, w))
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale

def nms(boxes, scores, iou_thres=0.3):
    """ simple non-max suppression on [y1,x1,y2,x2] """
    # If boxes is None or empty, bail out immediately
    if boxes is None or len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        yy1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        xx1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        yy2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        xx2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        inter = np.maximum(0, yy2-yy1) * np.maximum(0, xx2-xx1)
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_j = (boxes[idxs[1:],2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        idxs = idxs[1:][np.less_equal(iou, iou_thres)]
    return keep

def detect_icons(rgb_image, threshold=0.85, debug=False, max_hits_per_template=200):
    """Return list of (bbox, class_id, score) for each icon detected in the RGB image.
    Optimised: if the input image is large, work on a down-sampled copy and scale boxes back up."""

    t_total = time.time() if debug else None

    gray_full = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edge_full = cv2.Canny(gray_full, 50, 150)

    # Reduce size for faster template matching if needed (longest side > 1200px)
    edge, scale = _resize_image_keep_ratio(edge_full, max_dim=1200)

    detections = []

    for stem, tmpl in TEMPLATES.items():
        th, tw = TEMPLATE_SIZES[stem]

        # Skip very large templates relative to current image
        if th > edge.shape[0] or tw > edge.shape[1]:
            continue

        # Resize template to same scale variants and perform matching
        for sv in SCALE_VARIANTS:
            combined_scale = scale * sv
            h_s = int(th * combined_scale)
            w_s = int(tw * combined_scale)
            if h_s < 3 or w_s < 3 or h_s > edge.shape[0] or w_s > edge.shape[1]:
                continue
            tmpl_scaled = cv2.resize(tmpl, (w_s, h_s), interpolation=cv2.INTER_AREA)

            if debug:
                t_start = time.time()

            heat = cv2.matchTemplate(edge, tmpl_scaled, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(heat >= threshold)

            # If too many matches, keep only top-N highest scores to avoid huge arrays
            if ys.size > max_hits_per_template:
                scores_flat = heat[ys, xs]
                top_idx = np.argpartition(-scores_flat, max_hits_per_template)[:max_hits_per_template]
                ys = ys[top_idx]
                xs = xs[top_idx]
                if debug and scores_flat.size > max_hits_per_template:
                    print(f"Template '{stem}' sv{sv:.2f} had {scores_flat.size} hits, trimmed to {max_hits_per_template}")

            if debug and (ys.size > 0):
                print(f"Template '{stem}' matched {len(ys)} times (scale={combined_scale:.2f}) in {time.time()-t_start:.2f}s")

            # Convert to bounding boxes in original-scale coordinates
            for (x_s, y_s) in zip(xs, ys):
                y_full = y_s / combined_scale
                x_full = x_s / combined_scale
                h_full = th / combined_scale
                w_full = tw / combined_scale
                # Keep reasonable icon sizes (tuneable)
                if 10 <= w_full <= 150 and 10 <= h_full <= 150:
                    bbox = [float(y_full), float(x_full), float(y_full + h_full), float(x_full + w_full)]
                    score = float(heat[y_s, x_s])
                    detections.append((bbox, CLASS_MAP[stem], score))

    # NMS per class
    final = []
    for cid in set(d[1] for d in detections):
        boxes = [d[0] for d in detections if d[1] == cid]
        scores = [d[2] for d in detections if d[1] == cid]
        keep_idx = nms(boxes, scores)
        for k in keep_idx:
            final.append((boxes[k], cid, scores[k]))

    if debug:
        print(f"detect_icons: found {len(final)} detections in {time.time()-t_total:.2f}s (scale factor {scale:.2f})")

    return final

# Debug: show loaded template names once at import
print("Loaded icon templates:", list(TEMPLATES.keys())[:20], "... total", len(TEMPLATES))

# ------------------
# ORB feature setup
# ------------------

# Create ORB detector once (fast binary descriptors)
_ORB = cv2.ORB_create(500)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# For each retained template build keypoints & descriptors (use edge image for robustness)
TEMPLATE_KP = {}
TEMPLATE_DES = {}
for name, edge_img in list(TEMPLATES.items()):
    # Retrieve corresponding raw grayscale image size info by reversing name mapping
    # If name has suffixes, base template_raw may not match size; recompute kp on edge_img itself
    gray_for_kp = edge_img if edge_img.ndim == 2 else cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
    kp, des = _ORB.detectAndCompute(gray_for_kp, None)
    if des is None or len(kp) < 2:
        if DEBUG_LOAD:
            print(f"Remove {name}: only {0 if des is None else len(kp)} keypoints")
        # Too few features – remove template to avoid wasted work
        TEMPLATES.pop(name)
        continue
    TEMPLATE_KP[name] = kp
    TEMPLATE_DES[name] = des

    if DEBUG_LOAD:
        print(f"Keep {name}: size {edge_img.shape[1]}x{edge_img.shape[0]}, edge density {np.count_nonzero(edge_img)/(edge_img.shape[0]*edge_img.shape[1]):.2f}")

def detect_icons_orb(rgb_image, debug=False, min_inliers=5):
    """Detect camera/sensor icons using ORB keypoint matching & homography.

    Returns list of (bbox, class_id, score). bbox = [y1,x1,y2,x2].
    score is the inlier count.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    kp_img, des_img = _ORB.detectAndCompute(gray, None)
    if des_img is None or len(kp_img) < 10:
        if debug:
            print("ORB: too few keypoints in image – skipping icon detection")
        return []

    if debug:
        print(f"ORB: image has {len(kp_img)} keypoints")

    detections = []

    for name in TEMPLATES.keys():
        kp_t = TEMPLATE_KP[name]
        des_t = TEMPLATE_DES[name]

        matches = _BF.match(des_t, des_img)
        if debug:
            print(f"Template {name}: {len(matches)} raw matches")
        if len(matches) < min_inliers:
            continue

        # Take top 50 matches for speed
        matches = sorted(matches, key=lambda m: m.distance)[:50]
        # At least 40% of the matches must be within top-N to ensure robustness
        if len(matches) < 15:
            continue

        if debug:
            print(f"Template {name}: {len(matches)} raw matches")

        pts_t = np.float32([kp_t[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_i = np.float32([kp_img[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_t, pts_i, cv2.RANSAC, 5.0)
        if H is None:
            if debug:
                print(f"Template {name}: homography failed")
            continue
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < min_inliers or inliers < 0.25 * len(matches):
            if debug:
                print(f"Template {name}: only {inliers} inliers – rejected")
            continue

        if debug:
            print(f"Template '{name}' ORB inliers={inliers}, bbox=({int(pts_t[:, 0, 0].min())}x{int(pts_t[:, 0, 1].min())}, {int(pts_t[:, 0, 0].max())}x{int(pts_t[:, 0, 1].max())})")

        # Map template corners to image to get bbox
        h_t, w_t = TEMPLATES[name].shape[:2]
        corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        xs = proj[:, 0, 0]
        ys = proj[:, 0, 1]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # Basic sanity-check size
        w_box, h_box = x2 - x1, y2 - y1
        if w_box < 10 or h_box < 10 or w_box > 200 or h_box > 200:
            continue

        cid = CLASS_MAP.get(name, 4)
        detections.append(([float(y1), float(x1), float(y2), float(x2)], cid, float(inliers)))

    return detections 

def detect_icons_shape(rgb_image, debug=False, min_area=25, max_area=400, min_ar=0.6, max_ar=1.4):
    """Detects small, roughly-square/circular contours as icon candidates (template-free). Returns (bbox, class_id, score=1.0)."""
    import cv2
    import numpy as np
    
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale detection to handle different symbol sizes
    scales = [1.0, 1.5, 2.0]  # Try larger scales to better detect small symbols
    detections = []
    
    for scale in scales:
        if scale != 1.0:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            scaled = gray
            
        # Use simple thresholding first to find dark symbols
        _, binary = cv2.threshold(scaled, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Also try adaptive thresholding
        binary2 = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
                                      
        # Combine both thresholding results
        binary = cv2.bitwise_or(binary, binary2)
        
        # Clean up noise while preserving small circles
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours on both binary images
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Scale back coordinates
            cnt = (cnt / scale).astype(np.float32)
            
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
                
            # Get bounding box and check aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h > 0 else 0
            if not (min_ar <= ar <= max_ar):
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < 0.5:  # More lenient circularity for small symbols
                continue
                
            # Check if there's content inside (numbers)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt.astype(np.int32)], -1, 255, -1)
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            mean_val = cv2.mean(roi, mask=mask)[0]
            if mean_val > 240 or mean_val < 50:  # Must have some contrast inside
                continue
                
            # Convert to MRCNN format [y1, x1, y2, x2]
            bbox = [float(y), float(x), float(y + h), float(x + w)]
            detections.append((bbox, 4, 1.0))  # class 4 = camera
            
            if debug:
                print(f"Found camera at {bbox} (area={area:.1f}, ar={ar:.2f}, circ={circularity:.2f}, val={mean_val:.1f})")
    
    # Apply NMS to remove overlaps
    if detections:
        boxes = np.array([d[0] for d in detections])
        scores = np.array([d[2] for d in detections])
        keep = nms(boxes, scores, iou_thres=0.3)
        detections = [detections[i] for i in keep]
    
    if debug:
        print(f"detect_icons_shape: found {len(detections)} camera symbols")
    
    return detections 