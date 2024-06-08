import asyncio
import serial
import websockets
import cv2
import configparser
import os

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

async def main():
    if SERIAL_MODE == 1:
        arduino = serial.Serial( UART_PORT, UART_BAUDRATE, timeout=1.0)
        arduino.reset_input_buffer()
        print("Serial Initialized")

    cap = cv2.VideoCapture(CAMERA_PORT)
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            ret, frame = cap.read()
            _, img_encoded = cv2.imencode('.jpg', frame)
            data = img_encoded.tobytes()
            await websocket.send(data)
            response = await websocket.recv()
            x, found = response.split(',')
            x, found = int(x), bool(found)

            # Send current_position_x and current_position_y to Arduino continuously
            if SERIAL_MODE == 1 and found:
                arduino.write(f"{x}, -1\n".encode('utf-8'))

# Check if the script is run as the main program
if __name__ == "__main__":
    asyncio.run(main())