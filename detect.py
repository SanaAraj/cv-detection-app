from pathlib import Path
import uuid
import time
from ultralytics import YOLO
import cv2

model = YOLO("yolo11s.pt")

CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5
MAX_SIZE = 1920

CLASS_COLORS = [
    (255, 107, 107), (78, 205, 196), (69, 183, 209), (150, 206, 180),
    (255, 238, 173), (255, 204, 92), (254, 127, 45), (171, 135, 255),
    (119, 139, 235), (231, 127, 103), (207, 186, 240), (158, 224, 158),
]

def get_class_color(class_id):
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]

def run_detection(image_path: str, output_dir: str = "results") -> dict:
    Path(output_dir).mkdir(exist_ok=True)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(image_path, img)

    start_time = time.time()
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=False)[0]
    inference_time = time.time() - start_time

    detections = []
    class_set = set()
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        class_set.add(class_name)
        color = get_class_color(class_id)
        detections.append({
            "class": class_name,
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),
            "color": f"rgb({color[0]},{color[1]},{color[2]})"
        })

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class']} {det['confidence']:.2f}"
        rgb = det["color"][4:-1].split(",")
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    ext = Path(image_path).suffix
    output_filename = f"detected_{uuid.uuid4()}{ext}"
    output_path = str(Path(output_dir) / output_filename)
    cv2.imwrite(output_path, img)

    return {
        "detections": detections,
        "output_filename": output_filename,
        "stats": {
            "object_count": len(detections),
            "class_count": len(class_set),
            "inference_time": round(inference_time, 2)
        }
    }


if __name__ == "__main__":
    import urllib.request

    test_url = "https://ultralytics.com/images/bus.jpg"
    test_image = "test_bus.jpg"
    urllib.request.urlretrieve(test_url, test_image)

    result = run_detection(test_image)

    print(f"\nDetected {result['stats']['object_count']} objects across {result['stats']['class_count']} classes:")
    for d in result["detections"]:
        print(f"  - {d['class']}: {d['confidence']:.2%}")
    print(f"\nInference time: {result['stats']['inference_time']}s")
    print(f"Annotated image saved to: results/{result['output_filename']}")

    Path(test_image).unlink()
