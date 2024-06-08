import time
import asyncio
import serial
import websockets
import cv2
import configparser
import os
import json
import base64
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <parameter>")
    sys.exit(1)

# Get the parameter
param = sys.argv[1]

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

if SERIAL_MODE == 1:
    arduino = serial.Serial( UART_PORT, UART_BAUDRATE, timeout=1.0)
    arduino.reset_input_buffer()
    print("Serial Initialized")

cap = cv2.VideoCapture(CAMERA_PORT)

async def loop():

    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            ret, frame = cap.read()
            _, img_encoded = cv2.imencode('.jpg', frame)
            data = img_encoded.tobytes()
            array_base64 = base64.b64encode(data).decode('utf-8')

            payload = {
                'name': param,
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

if __name__ == '__main__':
    asyncio.run(loop())