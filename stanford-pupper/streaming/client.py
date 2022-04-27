import socket
import pickle
import time
import numpy as np
import torch
import torch.nn as nn


HEADERSIZE = 10
id = 0
data_obs = torch.zeros(48)
policy = nn.Sequential(nn.Linear(48, 512),
                       nn.ELU(),
                       nn.Linear(512, 256),
                       nn.ELU(),
                       nn.Linear(256, 128),
                       nn.ELU(),
                       nn.Linear(128, 12))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('192.168.100.163', 1245))

full_msg = b''
new_msg = True
wait_response=False

start_time = 0
end_time = 0

times = []

while True:

    if not wait_response:
        id += 1
        data_obs = torch.Tensor(data_obs)
        ac_data = policy(data_obs)#np.random.normal(size=(3,4))
        msg = pickle.dumps({'id':id, 'command':ac_data})
        print(f'-> [{id}] ac data send.')
        msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
        s.sendall(msg)
        wait_response = True
        start_time = time.time()

    else:
        while True:
            msg = s.recv(16)
            if new_msg:
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                reply = True
                data = pickle.loads(full_msg[HEADERSIZE:])
                data_id = data['id']
                data_obs = data['obs']
                end_time = time.time()
                new_msg = True
                full_msg = b""
                wait_response = False
                tmp = round(end_time - start_time, 5)
                print(f"@ Reply Received [{data_id}]:")
                print(f"Ellapsed time: {tmp} s.")
                times.append(tmp)
                break
    if len(times) >= 10000:
        print(f"MEAN TIME RESPONSE: {round(np.mean(times),5)} s | {round(1/np.mean(times),3)} hz")
        s.close()
        break
