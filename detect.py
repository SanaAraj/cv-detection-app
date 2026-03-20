from pathlib import Path
import uuid
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
MAX_SIZE = 1920

def run_detection(image_path: str, output_dir: str = "results") -> tuple[list[dict], str]:
    Path(output_dir).mkdir(exist_ok=True)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(image_path, img)

    results = model(image_path)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "class": results.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ext = Path(image_path).suffix
    output_filename = f"detected_{uuid.uuid4()}{ext}"
    output_path = str(Path(output_dir) / output_filename)
    cv2.imwrite(output_path, img)

    return detections, output_filename


if __name__ == "__main__":
    import urllib.request

    test_url = "https://ultralytics.com/images/bus.jpg"
    test_image = "test_bus.jpg"
    urllib.request.urlretrieve(test_url, test_image)

    detections, output_file = run_detection(test_image)

    print(f"\nDetected {len(detections)} objects:")
    for d in detections:
        print(f"  - {d['class']}: {d['confidence']:.2%}")
    print(f"\nAnnotated image saved to: results/{output_file}")

    Path(test_image).unlink()
