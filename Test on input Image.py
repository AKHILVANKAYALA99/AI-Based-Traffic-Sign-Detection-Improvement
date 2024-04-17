from ultralytics import YOLO
model = YOLO(r"#path to your best.pt file")
results = model(source=r"#Path to your Testing image", show=True, save=True)


for r in results:
    print(r.boxes.xyxy)