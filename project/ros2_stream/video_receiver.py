import cv2
import socket
import struct
import pickle

PI_HOST = "ubuntu-desktop.local"
PORT    = 9999

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PI_HOST, PORT))
print("Connected!")

data = b''
payload_size = struct.calcsize('>L')

while True:
    while len(data) < payload_size:
        data += client.recv(65536)

    msg_size = struct.unpack('>L', data[:payload_size])[0]
    data = data[payload_size:]

    while len(data) < msg_size:
        data += client.recv(65536)

    enc1, enc2 = pickle.loads(data[:msg_size])
    data = data[msg_size:]

    frame1 = cv2.imdecode(enc1, cv2.IMREAD_COLOR)
    frame2 = cv2.imdecode(enc2, cv2.IMREAD_COLOR)

    cv2.imshow("Top",  frame1)
    cv2.imshow("Side", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client.close()
cv2.destroyAllWindows()