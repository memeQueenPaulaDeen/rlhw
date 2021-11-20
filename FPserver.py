import array
import socket
import sys
import numpy as np

import cv2

if __name__ == '__main__':

    def reciveMessage(conn,BUFFER_SIZE=1024):
        data = conn.recv(BUFFER_SIZE)
        data = array.array('B', data)
        t = bytes(data).decode()
        if t[0] == 'l':
            connection.sendall("expecting to receive loc".encode())
            dataIn = conn.recv(BUFFER_SIZE)
            print('recived' + str(bytes(dataIn)))
        elif t[0] == 'p':
            size = int(t.split()[1])
            resp = 'expecting image of size ' + str(size)
            #print(resp)
            connection.sendall(resp.encode())
            picIn = conn.recv(size)
            picIn= np.fromstring(picIn, np.uint8)
            img_np = cv2.imdecode(picIn, flags=1)
            cv2.imshow('test',img_np)
            cv2.waitKey(1)
            #cv2.destroyAllWindows()




    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 10000)
    print('starting up on %s port %s' % server_address)
    sock.bind(server_address)
    # Listen for incoming connections
    sock.listen(1)
    print('waiting for a connection')
    connection, client_address = sock.accept()
    print("sucsessfull conection from " + str(client_address))

    while True:

        reciveMessage(connection)

