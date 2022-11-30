
# -*- coding: utf-8 -*-
"""
TCP 

Receive Raw Data From Verasonics 
    format: int16
    width: depends on probe (e.g. L11 4v has 128 channels)
    height: depends on sample mode (e.g. BS100BW provides 4096/2 samples per acquisition)

Return intensity parameter AND na_transmit as soon as data are received
    format: 2 elements as double 
    
Return Beamformed image to flask/webAPP
    format:JPEG
    
"""

from flask import Flask, render_template, Response, request, session
import cv2
import sys
import numpy

import socket
import numpy as np
import signal
import sys
import time
import array
import matplotlib.pyplot as plt

from predict import get_rt_model, load_saved_model
from demo_setup import get_models
import threading
import functools
import struct
import scipy.io
from datetime import datetime
from utils.video import FPS_counter, Scan_converter

#FUNCTIONS
def sigint_handler(signal, frame):
    print ('KeyboardInterrupt is caught')
    connection.close()
    save()
    # Close  
    sys.exit(0)
    
    
signal.signal(signal.SIGINT, sigint_handler)


global beamformer
beamformer = 'DAS'
global intensity
intensity = 2.0
global na_transmit
na_transmit = 1
global updateIntensity, updateBeamformer, update_na_transmit
updateIntensity = False
updateBeamformer = False
update_na_transmit = False

global na_read
na_read = 1
returnToMatlabFreq = 1

global flag
flag = 0

DUMMY_MODE = False
global show_fps
show_fps = False


fps_counter = FPS_counter()
global scan_converter
scan_converter = Scan_converter(norm_mode='smoothnormal', env_mode='abs', buffer_size=30)

trcvElapsedTime = []
tproElapsedTime = []
tsbElapsedTime = []
treadElapsedTime = [] #INCLUDE receive and processing
tIQDeodulationElapsedTime = []
tBeamformerElapsedTime = []
tReadPreProcessElapsedTime = []
tPreProcessElapsedTime = []

host = ''  # client: 131.155.127.59
port_tcp = 30000 #TCP
server_address_tcp = (host, port_tcp) 
timeOut = 1
bufferSize = 65500

probe = 'L114'

if probe == 'S51':
    Nax = 1152
    Nel = 80
    config = 'configs/training/default_S51.yaml'
elif probe == 'L114':
    Nax = 1024
    Nel = 128
    config = 'configs/training/default.yaml'

# Model
#OLD FORMAT
# models_folder = 'trained_models/'
# model_name = '0911_1805_realtime_ABLE_1PW_MV'
# model, probe, grid = load_saved_model(models_folder+model_name)

#NEW FORMAT

model_dict, grid = get_models()

WIDTH = grid[:,:,0].max()-grid[:,:,0].min()
HEIGHT = grid[:,:,2].max()-grid[:,:,2].min()
ASPECT_RATIO = WIDTH/HEIGHT
SCALE = 2

aspect_fx = (grid.shape[0]/grid.shape[1])*ASPECT_RATIO


bytesPerElementSent = 1 #uint8
T = 2
bytesPerElementRead = 2 #int16

temp = 0

        


     
app = Flask(__name__)
app.secret_key = 'Secret'
app.config["SESSION_PERMANENT"] = False


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.setdefaulttimeout(timeOut)
#s.setblocking(False)
s.settimeout(timeOut)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bufferSize)    
s.bind(server_address_tcp)
s.listen(1)   
print('TCP server is running ...')

timeDisplay = []

import scipy.io

def save():
    diff = []
    for k in range(len(timeDisplay)-1):
        c = timeDisplay[k+1] - timeDisplay[k]
        diff.append(c.total_seconds())
    d = {"interFrameSeconds1": diff, "tBeamformerElapsedTime1": tBeamformerElapsedTime, 
         "tPreProcessElapsedTime1": tPreProcessElapsedTime, "tReadPreProcessElapsedTime1": tReadPreProcessElapsedTime,}
    scipy.io.savemat('L114v_na1_GUIflask_python_displayRF.mat', mdict=d, format='5')

def get_frame():
    
        connection = None

        
        while True:
            
            try:
                
                if (connection == None) and (not DUMMY_MODE):
                                  
                    print('Waiting for connection...')
                    try: 
                        # Open connection
                        connection, client_address = s.accept()
                        print('Connected by', client_address)
                    except:
                        img = cv2.imread('python/templates/nofeed.jpg')
                        imgencode=cv2.imencode('.jpg',img)[1]
                        stringData=imgencode.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            
                else:
                    global intensity, updateIntensity, update_na_transmit, na_transmit, flag, na_read

                    if DUMMY_MODE:
                        IQ = 2**9*np.random.rand(1, na_transmit, Nel, int(Nax/2), 2)
                    else:
                        
                        # SEND UPDATE AS SOON AS READING FINISHES
                        #startTimeSB = time.time()
                        startTimeREADPRO = time.time()

                        if updateIntensity:
                            print('Intensity: ', intensity)
                            updateIntensity = False
                            tsb_lst = [intensity]
                        else:
                            tsb_lst =[0]
                    
                        if update_na_transmit:
                            flag = returnToMatlabFreq
                            print('Update firing angles: ', na_transmit)
                            update_na_transmit = False
                            tsb_lst.append(na_transmit)
                            
                        else:
                            tsb_lst.append(0)
                            
                            
                        if flag != 0:
                            flag -= 1
                        elif flag == 0:
                            na_read = na_transmit
                            #print('Number of firing angles: ', na_read)
                    
                        
                        # READ as INT16
                        total = 0
                        signal = bytearray()
                                            
                        while total < na_read*Nax*Nel*bytesPerElementRead: # Ensure to receive the complete data
                            data = connection.recv(bufferSize) #For best match with hardware and network realities, the value of bufsize should be a relatively small power of 2, for example, 4096.
                            if not data: break
                            #elif data: 
                                #if total == 0: startTimeRCV = time.time()
                            total = total + len(data)
                            signal += data
                        
                        # UPDATE INTENSITY
                        tsb_lst = list(map(np.double, tsb_lst))    
                        tsb_bytes = bytearray(struct.pack('%sd' % len(tsb_lst), *tsb_lst))
                        connection.sendall(tsb_bytes)
                        
                        
                        startTimePRO = time.time()
                        dataToBeProcessed = array.array('h', signal)
                    
                        
                        dataToBeProcessed = np.reshape(dataToBeProcessed, (Nax*na_read, Nel), order='F')
                        RFData = np.array_split(dataToBeProcessed, na_read)
                        RFData = np.stack(RFData, axis=2)  # Nax, Nel, na_transmit
                        RFData = np.transpose(RFData, (2, 1, 0))
                        
                        
                        # FROM RF DATA TO IQ
                        I = []
                        Q = []
                        for i in zip(RFData[:, :, ::2], RFData[:, :, 1::2]):
                            I.append(i[0])
                            Q.append(- i[1])
                                                
                        I = np.reshape(I, (1, na_read, Nel, int(Nax/2), 1))
                        Q = np.reshape(Q, (1, na_read, Nel, int(Nax/2), 1)) #expected: 1, na, 128, 1024, 1
                        
                        IQ = np.concatenate([I, Q], axis=-1)
                    
                        executionTimePRO = (time.time() - startTimePRO)
                        tPreProcessElapsedTime.append(executionTimePRO)
                        
                        executionTimeREADPRO =  (time.time() -startTimeREADPRO)
                        tReadPreProcessElapsedTime.append(executionTimeREADPRO)
                    
                    #IQ[:,:,:,:1,:] = np.zeros((1,1,128,1,2))
                    
                    startTimeBF = time.time()
                    
                    
                    # Select correct model from dictionary
                    try:
                        if na_transmit == 11:
                            if beamformer == 'DAS':
                                model = model_dict['DAS_11PW']
                            elif beamformer == 'ABLE':
                                model = model_dict['ABLE_11PW']
                        else:
                            if beamformer == 'DAS':
                                model = model_dict['DAS_1PW']
                            elif beamformer == 'ABLE':
                                model = model_dict['ABLE_1PW']


                        BF = model(IQ) #shape: 1,128,256
        
                        # if beamformer == 'DAS':
                        #     BF = BF[1]
                        # else:
                        #     BF = BF[0]
        
                        BF = np.squeeze(BF).T    
                    
                        BF = scan_converter.convert(BF)
                        
                        # RFT = np.transpose(RFData, (0, 2, 1))
                        # BF = RFT[0, :, :]
        
                        executionTimeBF = (time.time() - startTimeBF)
                        #print(executionTimeBF)
                        tBeamformerElapsedTime.append(executionTimeBF)
                        
                        # WRITE 
                        timeDisplay.append(datetime.now())                                    
                        # Display the resulting frame
                        BF = cv2.resize(BF, dsize=(int(SCALE*grid.shape[1]*aspect_fx), int(SCALE*grid.shape[0])), fx=aspect_fx)

                        # fps counter
                        if show_fps:
                            BF = fps_counter.overlay(BF)
                        
                        imgencode=cv2.imencode('.jpg',BF)[1]
                        stringData=imgencode.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
                    except:
                        pass
                    
                          
            except:
                if connection != None:
                    print('TCP server closes...')
                    connection.close()
                    connection = None
                    save()
                    
                             
                    


@app.route('/')
def index():
    return render_template('indexJQUERY.html')
 
@app.route('/create_file', methods=['POST'])
def create_file():
    if request.method == 'POST':
                
        if request.form.get('beamformer') != None:    
            global beamformer, updateBeamformer
            updateBeamformer = True
            beamformer = request.form.get('beamformer')

        if request.form.get('norm_mode') != None:    
            global scan_converter
            norm_mode = request.form.get('norm_mode')
            setattr(scan_converter, 'norm_mode', norm_mode)
        
        if request.form.get('slide') != None:    
            global intensity, updateIntensity
            updateIntensity = True
            intensity = request.form.get('slide')

        if request.form.get('fps') != None:
            global show_fps
            show_fps = not show_fps
            
        if request.form.get('na') != None:
            global na_transmit, update_na_transmit
            update_na_transmit = True
            na_transmit = int(request.form.get('na'))
            print(na_transmit)
        
        return ('', 204)
    
    

#@app.route('/vid/')
#@app.route('/vid/<connection>') #video feed
@app.route('/vid/')
def vid():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
  
    
if __name__ == '__main__':
        
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
        
        
        
            
    
        