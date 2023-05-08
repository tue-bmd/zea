# -*- coding: utf-8 -*-
"""
LOOP TCP RCV INT16 - TSB INT16
"""

import socket
import matplotlib.pyplot as plt
import signal
import sys
import array
import numpy as np
import time
#import matlab
import struct
import scipy.io

#FUNCTIONS
def sigint_handler(signal, frame):
    print ('KeyboardInterrupt is caught')

    # Clean up the connection
    connection.close()

    # Close
    sys.exit(0)

###############################################################################
###############################################################################




signal.signal(signal.SIGINT, sigint_handler)


trcvElapsedTime = []
tproElapsedTime = []
tsbElapsedTime = []
treadElapsedTime = [] #INCLUDE receive and processing
trcvRate = []
tsbRate = []
treadRate = [] #INCLUDE receive and processing

host = ''  # client: 131.155.127.59
port_tcp = 30000 #TCP
bufferSize = 65536
#'2048', '4096', '8192', '16384', '32768', 65536, 131072

server_address_tcp = (host, port_tcp)
timeOut = 30
na = 1
L = 128*1024*na # 524288
T = L
readSize = 65507 #bytes (4096)


bytesPerElementSent = 2
bytesPerElementRead = 2 #int16
chunkSize = []
# Define your connection
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    socket.setdefaulttimeout(timeOut)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bufferSize)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bufferSize)
    s.bind(server_address_tcp)
    s.listen(1)


    # Open connection
    print('TCP server is running ...')
    connection, client_address = s.accept()
    #s.setblocking(0)


    with connection:

        print('Connected by', client_address)

        try:
            iteration = 0

            while True:

                iteration = iteration + 1
                total = 0
                signal = bytearray()

                while total < L*bytesPerElementRead: # Ensure to receive the complete data
                    data = connection.recv(readSize) #For best match with hardware and network realities, the value of bufsize should be a relatively small power of 2, for example, 4096.
                    if not data: break
                    #elif data:
                        #print('Data Chunk size: '+ str(len(data)) + 'bytes') # length signal
                        #chunkSize.append(len(data))
                        #if total == 0: startTimeRCV = time.time()
                    total = total + len(data)
                    signal += data
                #executionTimeRCV = (time.time() - startTimeRCV)
                print('Received data size: '+ str(total) + 'bytes') # length signal
                dataToBeProcessed = array.array('h', signal)
                print('Received data size: '+ str(len(dataToBeProcessed)) + 'elements') # length signal

                tsb_lst = dataToBeProcessed
                #print(tsb_lst)

                tsb_lst = list(map(np.int16, tsb_lst))
                tsb_bytes = bytearray(struct.pack('%sh' % len(tsb_lst), *tsb_lst))
                print('Sent data size: '+ str(len(tsb_bytes)) + 'bytes')
                connection.sendall(tsb_bytes)
        #except KeyboardInterrupt:
            #print("Caught keyboard interrupt, exiting")
        except:
            rcvBuff = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            sndBuff =s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            print('TCP server closes...')
            connection.close()
