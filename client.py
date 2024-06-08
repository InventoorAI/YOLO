import time
import asyncio
import serial
import websockets
import cv2
import configparser
import os
from quart import Quart, request, websocket

import json
import base64

config_file = "config.ini"

if not os.path.exists(config_file):
    print('Config file not exist')

parser = configparser.ConfigParser()
parser.read(config_file)

UART_PORT = parser.get('uart','port')
UART_BAUDRATE = int(parser.get('uart','baudrate'))

# 0 for disabling serial, 1 turn on serial
SERIAL_MODE = int(parser.get('mode', 'serial'))

CAMERA_PORT = int(parser.get('camera', 'port'))

stack = []

app = Quart(__name__)

@app.route('/command', methods=['POST'])
async def receive_command():
    data = await request.json

    if (len(stack) == 0):
        stack.append(data['to'])
        stack.append(data['name'])
        stack.append(data['from'])
        return "OK"
    else:
        return "OPERATING"

async def loop():
    if SERIAL_MODE == 1:
        arduino = serial.Serial( UART_PORT, UART_BAUDRATE, timeout=1.0)
        arduino.reset_input_buffer()
        print("Serial Initialized")

    cap = cv2.VideoCapture(CAMERA_PORT)
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            if len(stack) == 0:
                continue;

            ret, frame = cap.read()
            _, img_encoded = cv2.imencode('.jpg', frame)
            data = img_encoded.tobytes()
            array_base64 = base64.b64encode(data).decode('utf-8')

            payload = {
                'name': stack[-1],
                'image': array_base64,
            }

            await websocket.send(json.dumps(payload))

            response = await websocket.recv()
            x, found = response.split(', ')
            x = int(x)
            found = True if found == 'True' else False
            print(f"X: {x}, found: {found}")

            # Send current_position_x and current_position_y to Arduino continuously
            if SERIAL_MODE == 1 and found:
                arduino.write(f"{x}, -1\n".encode('utf-8'))

            await asyncio.sleep(0.1)  # Yield control to the event loop

async def main():
    task1 = asyncio.create_task(app.run_task(port=3000))
    task2 = asyncio.create_task(loop())
    await asyncio.gather(task1, task2)

if __name__ == '__main__':
    asyncio.run(main())

# Check if the script is run as the main program
# if __name__ == "__main__":
#     asyncio.run(main())