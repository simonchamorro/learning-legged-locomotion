import socket
import time
import pickle
import numpy as np


HEADERSIZE = 10

reply = False

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1245))
s.listen(5)

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")

    full_msg = b''
    new_msg = True

    while True:

        if not reply:
            msg = clientsocket.recv(16)
            if new_msg:
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            #print(f"full message length: {msglen}")

            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                reply = True
                data = pickle.loads(full_msg[HEADERSIZE:])
                data_id = data['id']
                data_cmd = data['command']
                print(f"@ Command received [{data_id}]:")
                #print(data_cmd)
                new_msg = True
                full_msg = b""

                # Send back the state
                obs_data = np.random.normal(size=(48))
                msg = pickle.dumps({'id':data_id, 'obs':obs_data})
                msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
                print(f'->[{data_id}] Reply send.')
                clientsocket.sendall(msg)
                reply = False
