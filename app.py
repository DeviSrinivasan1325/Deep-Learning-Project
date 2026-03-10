from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import math, base64, cv2, numpy as np

app   = Flask(__name__, static_folder="static")
model = YOLO("runs/detect/train/weights/best.pt")
print("✅ Model loaded!")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()

    # Decode image
    img_bytes = base64.b64decode(data["image"].split(",")[1])
    img       = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w      = img.shape[:2]

    # Run model — fixed confidence 0.25, NMS 0.5
    results = model(img, conf=0.25, iou=0.5)

    child_center = None
    danger_center = None
    child_box     = None
    danger_box    = None
    detections    = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf  = float(box.conf)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx, cy = (x1+x2)/2, (y1+y2)/2

            detections.append({"label": label, "conf": round(conf,2),
                                "x1":x1,"y1":y1,"x2":x2,"y2":y2,"cx":cx,"cy":cy})

            if label == "child":
                child_center = (cx, cy)
                child_box    = (x1, y1, x2, y2)
            if label in ["knife","scissors"]:
                danger_center = (cx, cy)
                danger_box    = (x1, y1, x2, y2)

    # Alert logic
    alert        = False
    alert_reason = ""
    distance     = None
    edge_gap     = None

    if child_center and danger_center:
        # Center distance
        distance = math.sqrt((child_center[0]-danger_center[0])**2 +
                             (child_center[1]-danger_center[1])**2)

        # Edge gap
        def get_gap(b1, b2):
            gx = max(0, max(b1[0],b2[0]) - min(b1[2],b2[2]))
            gy = max(0, max(b1[1],b2[1]) - min(b1[3],b2[3]))
            return math.sqrt(gx**2 + gy**2)

        edge_gap   = get_gap(child_box, danger_box)
        diagonal   = math.sqrt(w**2 + h**2)
        threshold  = diagonal * 0.30   # 30% of image
        reach_zone = diagonal * 0.10   # 10% = reaching zone

        # Check overlap
        ox = child_box[0]<danger_box[2] and child_box[2]>danger_box[0]
        oy = child_box[1]<danger_box[3] and child_box[3]>danger_box[1]

        if ox and oy:
            alert        = True
            alert_reason = "touching"
        elif edge_gap < reach_zone:
            alert        = True
            alert_reason = "reaching"
        elif distance < threshold:
            alert        = True
            alert_reason = "near"

        print(f"dist={distance:.0f} gap={edge_gap:.0f} alert={alert} reason={alert_reason}")

    # Draw boxes
    for det in detections:
        x1,y1,x2,y2 = int(det["x1"]),int(det["y1"]),int(det["x2"]),int(det["y2"])
        color = (46,213,115) if det["label"]=="child" else (71,87,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        txt = f'{det["label"]} {det["conf"]:.0%}'
        cv2.rectangle(img,(x1,y1-22),(x1+len(txt)*9,y1),color,-1)
        cv2.putText(img,txt,(x1+4,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1)

    # Draw line
    if child_center and danger_center:
        lc = (0,71,255) if alert else (46,213,115)
        cv2.line(img,(int(child_center[0]),int(child_center[1])),
                     (int(danger_center[0]),int(danger_center[1])),lc,2)
        mx = int((child_center[0]+danger_center[0])/2)
        my = int((child_center[1]+danger_center[1])/2)
        cv2.putText(img,f"{distance:.0f}px",(mx+4,my-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,lc,1)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,90])
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return jsonify({
        "detections":   detections,
        "child_found":  child_center is not None,
        "danger_found": danger_center is not None,
        "distance":     round(distance,1) if distance else None,
        "edge_gap":     round(edge_gap,1) if edge_gap else None,
        "alert":        alert,
        "alert_reason": alert_reason,
        "result_image": img_b64
    })

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)
