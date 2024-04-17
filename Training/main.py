from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data = r'V:\Hexagon\Traffic_Signs\data.yaml',
    imgsz = 640,
    epochs = 500,
    batch = 10,
    name = 'yolov8n_custom',
    augment=True
)



