from ultralytics import YOLO

model = YOLO('./best_s.pt')

model.predict(source=1, show=True)
