# LICANCE-PLATE--DECODELAB

PROJECT 4: LICENSE PLATE DETECTION + OCR


Author: [Tahleel Fatima Iqbal]
For: Decode Lab - Image or Text Recognition (Basic)
Date: April 2026

DESCRIPTION:
This project detects license plates in images/videos using YOLOv8 
and reads the text using EasyOCR.

HOW TO RUN:
1. Install: pip install ultralytics opencv-python easyocr
2. Run Image: python app.py --source sample.jpg
3. Run Video: python app.py --source sample.mp4
4. Output: Saved in runs/detect/ folder

"""

from ultralytics import YOLO
import easyocr
import cv2
import argparse

def main():
    # 1. Arguments lo
    parser = argparse.ArgumentParser(description='License Plate Detection and Text Recognition')
    parser.add_argument('--source', type=str, required=True, help='Path to image or video file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    print("="*50)
    print("LICENSE PLATE DETECTOR")
    print("Author: [Your Name] | Decode Lab Project")
    print("="*50)

    # 2. Model load karo
    print("Loading Model: best.pt")
    model = YOLO('best.pt') # best.pt isi folder me honi chahiye
    reader = easyocr.Reader(['en'], gpu=False) # CPU

    # 3. Prediction karo
    print(f"Processing: {args.source}\n")
    results = model.predict(source=args.source, save
