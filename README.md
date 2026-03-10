# Child Safety Hazard Detection System 

An AI-powered deep learning object detection system that detects 
when a child is near a dangerous object like a knife or scissors 
and triggers a real-time alert.

---

## Project Overview

This project uses a custom trained YOLOv8 model to detect children 
and dangerous objects in images. When a child is detected near a 
dangerous object, the system calculates the distance between them 
and triggers one of three alert levels.

---

## Alert Levels

| Level | Description |
|-------|-------------|
| Near | Child and object within 30% of image diagonal |
| Reaching | Edge gap within 10% of image size |
| Touching | Bounding boxes overlapping |

---

## How It Works

1. Input image is passed to the YOLOv8 model
2. Model detects child, knife or scissors
3. Edge gap and distance is calculated between detections
4. Alert level is determined based on proximity
5. Annotated output image is returned with bounding boxes

---

## Tech Stack

- **YOLOv8** — Custom Object Detection Model
- **Python** — Core Programming Language
- **OpenCV** — Image Processing & Annotation
- **Flask** — Web App & REST API Deployment
- **NumPy** — Numerical Operations

---

## Dataset

Custom dataset with 3 classes:
- child
- knife
- scissors

---

## How to Run

### 1. Clone the Repository
git clone https://github.com/DeviSrinivasan1325/child-safety-detection.git
cd child-safety-detection

### 2. Install Requirements
pip install -r requirements.txt

### 3. Run the Flask App
python app.py

### 4. Open in Browser
http://localhost:5000

---

## Project Structure

├── app.py                  # Flask web application
├── dataset.yaml            # Dataset configuration
├── requirements.txt        # Dependencies
├── static/
│   └── index.html         # Frontend UI
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt # Trained model weights
└── test_images/            # Sample test images

---

## Test Images

| Image | Detection | Alert |
|-------|-----------|-------|
| Baby.png | Child only | ✅ No alert |
| Baby_with_knife.png | Child + Knife | ⚠️ Alert |
| Baby_with_scissor.jpg | Child + Scissors | ⚠️ Alert |

---

## Limitations

- Knife sometimes detected as scissors due to small dataset size
- Can be improved with more training data
- Currently supports image input only

---

## Future Improvements

- [ ] Add more dangerous object classes
- [ ] Support live webcam feed
- [ ] Expand dataset for better accuracy
- [ ] Mobile app deployment

---

## Author

Devi Srinivasan
Aspiring AI & Data Science Enthusiast


## 📄 License
MIT License
