from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import base64

from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('./best_s.pt')  # pretrained YOLOv8n model

def get_x(target, frame):
    frame = cv2.resize(frame, (480, 640), interpolation=cv2.INTER_AREA)

    results = model(frame, show=False)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()

    x = 0
    found = False
    mx = 0

    # Iterate through the results
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        confidence = conf
        detected_class = int(cls)
        name = results[0].names[detected_class]

        if (confidence < 0.6):
            continue
            
        if (name == target and x2-x1 > mx):
            found = True
            x = (x1 + x2) // 2
            mx = x2-x1

    x = 240 - x
    return x, found

def count_objects(frame):
    frame = cv2.resize(frame, (480, 640), interpolation=cv2.INTER_AREA)

    results = model(frame, verbose=False, show=False)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()

    A = dict()
    B = dict()

    # Iterate through the results
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        confidence = conf
        detected_class = int(cls)
        name = results[0].names[detected_class]

        if (confidence < 0.5):
            continue

        color = (0, 0, 255)
        thickness = 2

        # Draw the rectangle on the image
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{name} - {confidence:.2f}"

        label_position = (x1, y1 - 10)  # Position above the top-left corner

        # Define the font, font scale, and color for the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        label_color = (0, 0, 255)  # White color
        label_thickness = 1

        # Add the label to the image
        cv2.putText(frame, label, label_position, font, font_scale, label_color, label_thickness, cv2.LINE_AA)
    
        midpoint = (x1+x2)//2

        if (midpoint < 240):
            if name not in A:
                A[name] = 0
            A[name] += 1
        else:
            if name not in B:
                B[name] = 0
            B[name] += 1
 
    data = []

    for key, value in A.items():
        data.append({
            "name": key,
            "quantity": value,
            "site": "Site A"
        })
 
    for key, value in B.items():
        data.append({
            "name": key,
            "frequency": value,
            "site": "Site B"
        })

    return data, frame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Process the image (e.g., show it, save it, etc.)
            # cv2.imshow('Received Image', img)
            # cv2.waitKey(1)
            x, found = get_x("blue-ball", img)
            await websocket.send_text(f"{x}, {found}")
        except Exception as e:
            print(f"Connection closed: {e}")
            break

@app.post("/count")
async def count(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    data, frame = count_objects(img)

    frame = cv2.resize(frame, (400, 250), interpolation=cv2.INTER_AREA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame)

    # Save the image to a BytesIO object
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Encode the image to base64
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return {
        "data": data,
        "base64": img_base64
    }

@app.get("/")
async def hello_world():
    return "Hello World"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)