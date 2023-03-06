
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

import array
import collections
import logging
import socket
import struct
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import scipy.io
from demo_setup import get_models
from flask import Flask, Response, render_template, request
from futures3.thread import ThreadPoolExecutor

from usbmd.utils.video import FPS_counter, Scan_converter


## HELPER FUNCTIONS
def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
##

# Set logger
if debugger_is_active():
    logging.basicConfig(level=logging.DEBUG)


class WebServer:
    """Webserver class that interfaces with the Verasonics, and handles data processing"""
    def __init__(
        self,
        host = '',
        tcp_port = 30000,
        time_out = 1,
        buffer_size = 2**16,
        probe = 'L114'):
        """_summary_

        Args:
            host (str, optional): Server adress. If empty we will host locally. Defaults to ''.
            tcp_port (int, optional): TCP port for Verasonics communication. Defaults to 30000.
            time_out (int, optional): . Defaults to 1.
            buffer_size (int, optional): Buffer size (bytes) of the websocket. Defaults to 2**16.
            probe (str, optional): Probe settings. Defaults to 'L114'.
            dummy_mode (bool, optional): If True, random test data is generated. Defaults to False.
        """

        ## Network settings
        self.host = host
        self.tcp_server_address = (host, tcp_port)
        self.time_out = 1
        self.buffer_size = 65500
        self.vera_socket = self.start_tcp_server(self.tcp_server_address, time_out, buffer_size)
        self.source = 'verasonics'

        ## Objects
        self.fps_counter = FPS_counter()
        self.scan_converter = Scan_converter(
            norm_mode='smoothnormal',
            env_mode='abs',
            img_buffer_size=30,
            max_buffer_size=30)

        #self.sr_model = keras.models.load_model('trained_models/SR02122022/generator.h5')
        self.model_dict, self.grid = get_models()


        ## State-parameters (perhaps move this to a separate class/state handler in the future?)
        self.bf_type = 'DAS' # Set beamformer type to DAS by default
        self.intensity = 2.0 # Transmit voltage
        self.na_transmit = 1 # Number of transmits (PWs/DWs)
        self.na_read = self.na_transmit
        self.update_intensity = False
        self.update_beamformer = False
        self.update_na_transmit = False
        self.flag = 0
        self.auto_update_intensity = False
        self.ref_amp = 80
        self.c = 1540 # Speed of sound

        # Probe settings
        if probe == 'S51':
            self.n_ax = 1152
            self.n_el = 80
        elif probe == 'L114':
            self.n_ax = 1152
            self.n_el = 128

        # Display Settings
        self.show_fps = False
        self.upscaling = False

        self.width = self.grid[:,:,0].max()-self.grid[:,:,0].min()
        self.height = self.grid[:,:,2].max()-self.grid[:,:,2].min()
        self.aspect_ratio = self.width/self.height
        self.aspect_fx = (self.grid.shape[0]/self.grid.shape[1])*self.aspect_ratio
        self.scaling = 3

        ## Buffers
        self.bf_display = 0 #BF_display
        self.bf_previous = 0

        # Benchmark variables
        self.beamformer_elapsed_time = []
        self.read_preprocess_elapsed_time = []
        self.update_elapsed_time = []
        self.time_display= []

        # Other (to be organized)
        self.meanAmp_history = []
        self.err_history = []
        self.hv_history = []
        self.returnToMatlabFreq = None
        self.bytesPerElementRead = None
        self.bytesPerElementSent = None
        self.numTunableParameters = None

    @staticmethod
    def start_tcp_server(tcp_server_address, time_out, buffer_size):
        "Function that starts the TCP server which listens for Verasonics data"
        vera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.setdefaulttimeout(time_out)
        vera_socket.settimeout(time_out)
        vera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
        vera_socket.bind(tcp_server_address)
        vera_socket.listen(1)
        logging.info('TCP server is running ...')
        return vera_socket


    def read_update(self, connection, buffer, terminate_signal):
        """Retrieves new data from the client"""
        while not terminate_signal.is_set(): #True:
            try:
                logging.debug('Start updating and reading')
                start_time_update = time.time()

                if self.source == 'dummy':
                    IQ = 2**9*np.random.rand(1, self.na_transmit, self.n_el, int(self.n_ax/2), 2)
                    buffer.append(IQ)
                else:
                    # SEND UPDATE AS SOON AS READING FINISHES
                    #startTimeSB = time.time()

                    if self.update_intensity:
                        #logging.debug('Intensity: %f', self.intensity)
                        self.update_intensity = False
                        tsb_lst = [self.intensity]
                    elif self.auto_update_intensity:
                        logging.debug('Entered updateIntensityAuto')

                        abs_signal = np.absolute(np.array(self.bf_previous))
                        meanAmp = np.mean(abs_signal)
                        self.meanAmp_history.append(meanAmp)
                        logging.debug('Mean RF amplitude: %f', meanAmp)
                        err = self.ref_amp - meanAmp
                        self.err_history.append(err)
                        logging.debug('Error in intensity: %f', err)

                        # PID CONTROLLER
                        # TODO: Implement this as a separate class
                        Kp = 0.05 # Proportional
                        Kd = 0 # Derivative
                        Ki = 0.001 # Integral
                        delta_hv = (Kd+Kp+Ki)*self.err_history[-1] + \
                                    (Ki-Kp-2*Kd)*self.err_history[-2] + \
                                    Kd*self.err_history[-3]

                        if len(self.hv_history) >= 1:
                            hv = self.hv_history[-1] + delta_hv
                        else:
                            hv = float(self.intensity) + delta_hv

                        #check value within range
                        hv = np.minimum(hv, 40)
                        hv = np.maximum(hv, 1.6)

                        self.hv_history.append(hv)
                        tsb_lst =[hv]
                    else:
                        tsb_lst =[0]

                    if self.update_na_transmit:

                        self.flag = self.returnToMatlabFreq
                        print('Update firing angles: ', self.na_transmit)
                        self.update_na_transmit = False
                        tsb_lst.append(self.na_transmit)
                        self.na_read = self.na_transmit
                    else:
                        tsb_lst.append(0)

                    if len(tsb_lst) != self.numTunableParameters:
                        print('Length of the to-sent-back (tsb) list different from expected')

                    # SEND UPDATE as DOUBLE
                    tsb_lst = list(map(np.double, tsb_lst))
                    tsb_bytes = bytearray(struct.pack(f'{len(tsb_lst)}d', *tsb_lst))
                    connection.sendall(tsb_bytes)

                    executionTimeUPD = time.time() - start_time_update
                    self.update_elapsed_time.append(executionTimeUPD)

                    ############################################
                    startTimeREADPRO = time.time()
                    # READ CHANNEL DATA as INT16
                    total = 0
                    signal = bytearray()

                    # Ensure to receive the complete data
                    while total < self.na_read*self.n_ax*self.n_el*self.bytesPerElementRead:
                        #For best match with hardware and network realities, the value of bufsize
                        #should be a relatively small power of 2, for example, 4096.
                        data = connection.recv(self.buffer_size)
                        if not data:
                            break
                        total = total + len(data)
                        signal += data

                    dataToBeProcessed = array.array('h', signal)
                    dataToBeProcessed = np.reshape(
                                                dataToBeProcessed,
                                                (self.n_ax*self.na_read, self.n_el),
                                                order='F')
                    RFData = np.array_split(dataToBeProcessed, self.na_read)
                    RFData = np.stack(RFData, axis=2)  # Nax, Nel, na_transmit
                    RFData = np.transpose(RFData, (2, 1, 0))

                    # FROM RF DATA TO IQ
                    I = []
                    Q = []
                    for i in zip(RFData[:, :, ::2], RFData[:, :, 1::2]):
                        I.append(i[1])
                        Q.append(i[0])

                    I = np.reshape(I, (1, self.na_read, self.n_el, int(self.n_ax/2), 1))
                    Q = np.reshape(Q, (1, self.na_read, self.n_el, int(self.n_ax/2), 1))

                    IQ = np.concatenate([I, Q], axis=-1)
                    buffer.append(IQ)

                    executionTimeREADPRO =  time.time() -startTimeREADPRO
                    self.read_preprocess_elapsed_time.append(executionTimeREADPRO)

            except:
                logging.debug('set terminate signal...')
                terminate_signal.set()  # signal writer that we are done

    def save(self):
        """Function that saves benchmark data to .mat file"""
        displayInterFramePeriod = []
        for k in range(len(self.time_display)-1):
            c = self.time_display[k+1] - self.time_display[k]
            displayInterFramePeriod.append(c.total_seconds())

        d = {
            f"""displayIFP_na{int(self.na_transmit)}
            _proc{self.bf_type}
            _autotun{str(self.auto_update_intensity)}""": displayInterFramePeriod,

            f"""tBeamformerElapsedTime_na{int(self.na_transmit)}
            _proc{self.bf_type}
            _autotun{str(self.auto_update_intensity)}""": self.beamformer_elapsed_time,

            f"""tUpdateElapsedTime_na{int(self.na_transmit)}
            _proc{self.bf_type}
            _autotun{str(self.auto_update_intensity)}""": self.update_elapsed_time,

            f"""tReadPreProcessElapsedTime_na{int(self.na_transmit)}
            _proc{self.bf_type}
            _autotun{str(self.auto_update_intensity)}""": self.read_preprocess_elapsed_time,
        }
        scipy.io.savemat(
            f"""L114v_na{int(self.na_transmit)}
            _proc{self.bf_type}
            _autotun{str(self.auto_update_intensity)}
            _pyt_{int(time.time())}.mat""",
            mdict=d)

    def process_data(self, buffer, terminate_signal):
        """Function that handles data processing (e.g. beamforming)"""
        while not terminate_signal.is_set():
            logging.debug('start processing')
            try:
                IQ = buffer.pop() #expected: 2, na, 128, 1024, 1
                buffer.clear()

                startTimeBF = time.time()

                # Select correct model from dictionary
                if self.bf_type == 'RAW':
                    def return_IQ(*args, **kwargs):
                        ix = int(np.floor(IQ.shape[1]/2))
                        return IQ[0,ix, :, :,0]
                    model = return_IQ
                elif IQ.shape[1] == 11:
                    if self.bf_type == 'DAS':
                        model = self.model_dict['DAS_11PW']
                    elif self.bf_type == 'ABLE':
                        model = self.model_dict['ABLE_11PW']
                elif IQ.shape[1] == 5:
                    if self.bf_type == 'DAS':
                        model = self.model_dict['DAS_5PW']
                    elif self.bf_type == 'ABLE':
                        model = self.model_dict['ABLE_5PW']
                elif IQ.shape[1] == 1:
                    if self.bf_type == 'DAS':
                        model = self.model_dict['DAS_1PW']
                    elif self.bf_type == 'ABLE':
                        model = self.model_dict['ABLE_1PW']

               
                c_input = np.expand_dims(self.c, (0,1))
                BF = model([c_input, IQ])
                BF = np.squeeze(BF)
                BF = self.scan_converter.convert(BF)

                # if self.upscaling:
                #     BF = np.expand_dims(BF,-1)
                #     BF = cv2.merge((BF, BF, BF))
                #     BF = np.expand_dims(BF, 0)
                #     BF = self.sr_model(BF/255)
                #     BF = np.squeeze(BF)*255

                self.time_display.append(datetime.now())
                #Display the resulting frame
                BF = cv2.resize(
                    BF,
                    dsize=(
                    int(self.scaling*self.grid.shape[1]*self.aspect_fx),
                    int(self.scaling*self.grid.shape[0])),
                    fx=self.scaling*self.aspect_fx
                    )

                # FPS counter
                if self.show_fps:
                    BF = self.fps_counter.overlay(BF)

                self.bf_display = BF
                executionTimeBF = time.time() - startTimeBF
                self.beamformer_elapsed_time.append(executionTimeBF)
            except:
                time.sleep(0.0001)  # wait for new data


    def open_connection(self):
        """Handles the opening and handshaking with the Verasonics"""
        # Open connection
        connection, client_address = self.vera_socket.accept()
        print('Connected by', client_address)
        # READ INITIALIZATION PARAMETERS
        total = 0
        msg = bytearray()

        while total < 7: # Ensure to receive the complete data
            #For best match with hardware and network realities,
            # the value of bufsize should be a relatively small power of 2,
            # for example, 4096.
            data = connection.recv(self.buffer_size)
            if not data:
                break
            total = total + len(data)
            msg += data

        initializationParameters = array.array('h', msg)

        self.n_ax = initializationParameters[0]
        self.n_el = initializationParameters[1]
        self.na_transmit = initializationParameters[2]
        self.na_read = self.na_transmit

        self.returnToMatlabFreq = initializationParameters[3]
        self.bytesPerElementSent = initializationParameters[4]
        self.bytesPerElementRead = initializationParameters[5]
        self.numTunableParameters = initializationParameters[6]

        # ACK INITIALIZATION COMPLETED
        ack = [1]
        ack_bytes = bytearray(struct.pack(f'{len(ack)}B', *ack))
        connection.sendall(ack_bytes)

        logging.debug('Initialization completed')

        return connection

    def get_frame(self):
        """Function that generates new image frames for the web interface"""
        connection = None
        while True:
            try:
                if (connection is None) and (self.source != 'dummy'):
                    print('Waiting for connection...')
                    try:
                        connection = self.open_connection()
                    except:
                        img = cv2.imread('usbmd/verasonics/webserver/templates/nofeed.jpg')
                        yield self.encode_img(img)

                else:
                    logging.debug('Entered the image formation loop.')

                    buffer = collections.deque([], maxlen=30)  # buffer for reading/writing
                    terminate_signal = threading.Event()  # shared signa

                    with ThreadPoolExecutor(2) as executor:
                        _ = executor.submit(self.read_update, connection, buffer, terminate_signal)
                        _ = executor.submit(self.process_data, buffer, terminate_signal)

                        while not terminate_signal.is_set():
                            try:
                                yield self.encode_img(self.bf_display)
                            except:
                                time.sleep(0.001)  # wait for new data

                    print('Saving results...')
                    #self.save()
                    connection = None
                    print(connection)

            except:
                if connection is not None:
                    print('TCP server closes...')
                    connection.close()
                    connection = None
                    print('Saving results...')
                    #self.save()

    @staticmethod
    def encode_img(img):
        """Function that encodes the input image to jpeg and prepares the HTML package"""
        imgencode=cv2.imencode('.jpg',img)[1]
        stringData=imgencode.tobytes()
        return (
            b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n'
            )

    def update_settings(self):
        """Function that updates setting states of the webserver class"""
        if request.method == 'POST':
            if request.form.get('beamformer') is not None:
                self.update_beamformer = True
                self.bf_type = request.form.get('beamformer')
                logging.debug(self.bf_type)

            if request.form.get('norm_mode') is not None:
                norm_mode = request.form.get('norm_mode')
                setattr(self.scan_converter, 'norm_mode', norm_mode)
                logging.debug(norm_mode)

            if request.form.get('na') is not None:
                self.save()
                self.beamformer_elapsed_time = []
                self.read_preprocess_elapsed_time = []
                self.update_elapsed_time = []
                self.time_display = []

                self.update_na_transmit = True
                self.na_transmit = int(request.form.get('na'))
                logging.debug(self.na_transmit)

            if request.form.get('intensityAuto') is not None:

                self.save()
                self.beamformer_elapsed_time = []
                self.read_preprocess_elapsed_time = []
                self.update_elapsed_time = []
                self.time_display = []

                self.auto_update_intensity = not self.auto_update_intensity
                logging.debug(self.auto_update_intensity)
                if self.auto_update_intensity:
                    self.meanAmp_history = []
                    self.err_history = [0,0]
                    self.hv_history = []

            if (request.form.get('slide_voltage') is not None) and not self.auto_update_intensity:
                self.update_intensity = True
                self.intensity = float(request.form.get('slide_voltage'))
                logging.debug(self.intensity)

            if request.form.get('slide_persistence') is not None:
                val_persistence = int(request.form.get('slide_persistence'))
                self.scan_converter.n_persistence = val_persistence
                logging.debug(val_persistence)
            
            if request.form.get('slide_alpha') is not None:
                self.scan_converter.alpha = float(request.form.get('slide_alpha'))
                logging.debug(self.scan_converter.alpha)
            
            if request.form.get('slide_sos') is not None:
                self.c = float(request.form.get('slide_sos'))
                logging.debug(self.c)

            if request.form.get('persistence_mode') is not None:
                mode = request.form.get('persistence_mode')
                self.scan_converter.persistence_mode = mode
                logging.debug(self.scan_converter.persistence_mode)

            if request.form.get('fps') is not None:
                self.show_fps = not self.show_fps
                logging.debug(self.show_fps)

            if request.form.get('upscaling') is not None:
                self.upscaling = not self.upscaling
                logging.debug(self.upscaling)

            if request.form.get('source') is not None:
                self.source = request.form.get('source')
                logging.debug(self.source)


        return ('', 204)


# Initialize Flask server
app = Flask(__name__)
app.secret_key = 'Secret'
app.config["SESSION_PERMANENT"] = False

@app.route('/')
def index():
    """Creates the HTML page"""
    return render_template('index.html')

@app.route('/create_file', methods=['POST'])
def create_file():
    """Function that updates server settings based on user input"""
    return web_server.update_settings()

@app.route('/vid/')
def vid():
    """ Video Feed"""
    return Response(web_server.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Initialize Verasonics webserver
web_server = WebServer()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=debugger_is_active(), use_reloader=False)
