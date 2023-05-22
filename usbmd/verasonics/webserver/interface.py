
""" Server module for handling cloud based ultrasound data processing (HOST) and communication with
the Verasonics Vantage system (CLIENT). This module contains:

- a Flask server that hostst a web application which can be used to visualize the
processed data and send commands to the Verasonics system.
- a class for handling ultrasound data processing and communication with the Verasonics system.

Usage:
- Start the server by running this module as a script. The webserver will be started on port 5000
of the host machine. The web application can be accessed locally via localhost:5000 or externally
via the external IP address of the host machine (e.g. 131.155.125.231:5000)
-On the Verasonics machine, run the setup script and start the modified VSX script, VSX_demo.m. Make
sure that the IPv4 address of the host machine is entered correctly in the setup script.
- By default the server will start listening on port 30000 for incoming data from the Verasonics.
Alternatively, one can select to generate dummy data (random noise) by selecting the 'Dummy data'
option in the web application. This can be helpfull for testing the web application without having
access to the Verasonics system.
- The configuration of the server is currently hard coded in the demo_setup.py module. This includes
the probe settings and beamforming parameters.

Authors: Beatrice Federici, Ben Luijten
"""

import array
import collections
import logging
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
import scipy.io
import tensorflow as tf
from demo_setup import get_models
from flask import Flask, Response, render_template, request
from futures3.thread import ThreadPoolExecutor

from usbmd.utils.video import FPS_counter, ScanConverterTF
from usbmd.verasonics.webserver.benchmarking import BenchmarkTool
from usbmd.verasonics.webserver.control import PIDController

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


# Set logger
if debugger_is_active():
    logging.basicConfig(level=logging.DEBUG)


class UltrasoundProcessingServer:
    """ Class for handling ultrasound data processing and communication with the Verasonics"""

    def __init__(
            self,
            host='',
            tcp_port=30000,
            time_out=0.5,
            buffer_size=2**16,
            probe='L114',
            saving=False):
        """_summary_

        Args:
            host (str, optional): Server adress. If empty we will host locally. Defaults to ''.
            tcp_port (int, optional): TCP port for Verasonics communication. Defaults to 30000.
            time_out (int, optional): . Defaults to 1.
            buffer_size (int, optional): Buffer size (bytes) of the websocket. Defaults to 2**16.
            probe (str, optional): Probe settings. Defaults to 'L114'.
            dummy_mode (bool, optional): If True, random test data is generated. Defaults to False.
            saving (bool, optional): If True, benchmark timings saved to disk. Defaults to False.
            This is currently working alongside the benchmarking tool.
        """

        # Network settings
        self.host = host
        self.tcp_server_address = (host, tcp_port)
        self.time_out = time_out
        self.buffer_size = 65500
        self.vera_socket = self.start_tcp_server(
            self.tcp_server_address, self.time_out, buffer_size)
        self.source = 'verasonics'
        self.saving = saving

        # self.sr_model = keras.models.load_model('trained_models/SR02122022/generator.h5')
        self.model_dict, self.grid = get_models()
        self.active_model = self.model_dict['DAS_1PW']

        # Objects
        self.fps_counter = FPS_counter()
        self.scan_converter = ScanConverterTF(
            self.grid,
            norm_mode='smoothnormal',
            env_mode='abs',
            img_buffer_size=30,
            max_buffer_size=30)

        # State-parameters (perhaps move this to a separate class/state handler in the future?)
        self.bf_type = 'DAS'  # Set beamformer type to DAS by default
        self.intensity = 2.0  # Transmit voltage
        self.na_transmit = 1  # Number of transmits (PWs/DWs)
        self.na_read = self.na_transmit
        self.update_intensity = False
        self.update_beamformer = False
        self.update_na_transmit = False
        self.flag = 0
        self.auto_update_intensity = False
        self.ref_amp = 80
        self.voltage_controller = PIDController(
            Kp=0.05,
            Ki=0,
            Kd=0.001,
            setpoint=self.ref_amp,
            min_val=1.6,
            max_val=40
        )

        self.c_ref = 1540  # Reference speed of sound
        self.c = 1540  # Speed of sound
        self.fs = 6.25e6  # Sampling frequency
        self.fc = 6.25e6  # Center frequency

        # Stores inputs for the beamformer model
        self.inputs = {}

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

        self.width = self.grid[:, :, 0].max()-self.grid[:, :, 0].min()
        self.height = self.grid[:, :, 2].max()-self.grid[:, :, 2].min()
        self.aspect_ratio = self.width/self.height
        self.aspect_fx = (
            self.grid.shape[0]/self.grid.shape[1])*self.aspect_ratio
        self.scaling = 3

        # Buffers
        self.bf_display = None  # BF_display
        self.bf_previous = 0

        # Benchmark variables
        self.benchmark_tool = BenchmarkTool(
            output_folder='benchmarks',
            benchmark_config='usbmd/verasonics/webserver/benchmark_config.yaml'
            )

        self.beamformer_elapsed_time = []
        self.read_preprocess_elapsed_time = []
        self.update_elapsed_time = []
        self.processing_clock = []
        self.display_clock = []
        self.reading_clock = []
        self.processing_id = []
        self.global_id = 0

        # Other (to be organized)
        self.meanAmp_history = []
        self.err_history = []
        self.hv_history = []
        self.returnToMatlabFreq = None
        self.bytesPerElementRead = None
        self.bytesPerElementSent = None
        self.numTunableParameters = None

        self.should_exit = False
        main_thread = threading.Thread(target=self.start_processing)
        main_thread.daemon = True
        main_thread.start()
        print('Main thread started')

        # Termination signal for child threads
        self.terminate_signal = threading.Event()



    @staticmethod
    def start_tcp_server(tcp_server_address, time_out, buffer_size):
        "Function that starts the TCP server which listens for Verasonics data"
        vera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.setdefaulttimeout(time_out)
        vera_socket.settimeout(time_out)
        vera_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
        vera_socket.bind(tcp_server_address)
        vera_socket.listen(1)
        logging.info('TCP server is running ...')
        return vera_socket

    @staticmethod
    def indicator(message):
        """ Waiting indicator for the console"""
        symbols = ['|', '/', '-', '\\']
        while True:
            for symbol in symbols:
                yield message + symbol

    @staticmethod
    def is_connection_open(connection):
        """Check if the given WebSocket connection is open."""
        return connection is not None and connection.fileno() != -1

    def start_processing(self):
        """Start the main processing loop."""
        print('Starting processing in background...')
        connection = None
        waiting_for_connection = self.indicator(
            'Waiting for connection to Verasonics ')

        while not self.should_exit:
            try:
                # Try to establish connection if none exists
                if not self.is_connection_open(connection) and self.source != 'dummy':
                    print(next(waiting_for_connection), flush=True, end='\r')
                    try:
                        connection = self.open_connection()
                        time.sleep(0.5)
                    except:
                        img = cv2.imread(
                            'usbmd/verasonics/webserver/templates/nofeed.jpg')
                        self.bf_display = img

                # Process data if connection is open
                if self.is_connection_open(connection) or self.source == 'dummy':
                    logging.debug('Entered the image formation loop.')
                    buffer = collections.deque([], maxlen=10)
                    with ThreadPoolExecutor() as executor:
                        executor.submit(self.read_update, connection,
                                        buffer, self.terminate_signal)
                        executor.submit(self.process_data, buffer,
                                        self.terminate_signal)
                    self.terminate_signal.clear()  # Reset the termination signal
            except:
                logging.exception('Exception in main processing loop.')

        # Close connection if it exists
        if connection:
            print('Closing connection...')
            connection.close()

    def prepare_inputs(self, buffer):
        """Function that prepares the inputs for the beamformer model"""
        # Prepare inputs for the beamforming model
        if buffer:
            buf = buffer.pop()
            buffer.clear()
        else:
            return None

        IQ = buf['data']
        frame_id = buf['frame_id']

        inputs = {}

        if self.bf_type == 'RAW':
            inputs['data'] = tf.convert_to_tensor(IQ, dtype=tf.float32)
        else:
            for inp in self.active_model.inputs:
                varname = inp.name.strip('_input')

                if varname == 'data':
                    inputs[varname] = tf.convert_to_tensor(IQ, dtype=inp.dtype)
                elif varname == 'grid':
                    inputs[varname] = tf.convert_to_tensor(
                        np.expand_dims(self.grid, 0),
                        dtype=inp.dtype
                    )
                else:
                    inputs[varname] = tf.convert_to_tensor(
                        np.expand_dims(getattr(self, varname), axis=(0, 1)),
                        dtype=inp.dtype
                    )

        return inputs, frame_id

    def read_update(self, connection, buffer, terminate_signal):
        """Retrieves new data from the client"""
        while not terminate_signal.is_set():
            if (self.source != 'dummy') and (connection is None):
                break
            try:
                logging.debug('Start updating and reading')
                start_time_update = time.perf_counter()

                if self.source == 'dummy':
                    IQ = np.random.randint(
                        low=-1000,
                        high=1000,
                        size=(1, self.na_transmit, self.n_el,
                              int(self.n_ax/2), 2),
                        dtype=np.int16
                    )
                    buf = {'data': IQ, 'frame_id': self.global_id}
                    self.global_id += 1
                    buffer.append(buf)

                elif connection.fileno() != -1:
                    # startTimeSB = time.perf_counter()

                    if self.update_intensity:
                        # logging.debug('Intensity: %f', self.intensity)
                        self.update_intensity = False
                        tsb_lst = [self.intensity]
                    elif self.auto_update_intensity:
                        logging.debug('Entered updateIntensityAuto')

                        abs_signal = np.absolute(np.array(self.bf_previous))
                        meanAmp = np.mean(abs_signal)
                        hv = self.voltage_controller.update(meanAmp)

                        self.hv_history.append(hv)
                        tsb_lst = [hv]
                    else:
                        tsb_lst = [0]

                    if self.update_na_transmit:
                        #self.flag = self.returnToMatlabFreq
                        print('Update firing angles: ', self.na_transmit)
                        self.update_na_transmit = False
                        tsb_lst.append(self.na_transmit)
                        self.na_read = self.na_transmit
                    else:
                        tsb_lst.append(0)

                    if self.update_beamformer:

                        matchcase = self.bf_type
                        if matchcase == 'RAW':
                            tsb_lst.append(1)
                        elif matchcase == 'DAS':
                            tsb_lst.append(2)
                        elif matchcase == 'ABLE':
                            tsb_lst.append(3)
                        else:
                            logging.error('Beamformer type does not match.')

                        self.update_beamformer = False
                    else:
                        tsb_lst.append(0)

                    if len(tsb_lst) != self.numTunableParameters:
                        print(
                            'Length of the to-sent-back (tsb) list different from expected')

                    # SEND UPDATE as DOUBLE
                    tsb_lst = list(map(np.double, tsb_lst))
                    tsb_bytes = bytearray(struct.pack(
                        f'{len(tsb_lst)}d', *tsb_lst))
                    print('going to send tsb now')
                    connection.sendall(tsb_bytes)
                    print('this is the tsb list: ', tsb_lst)

                    executionTimeUPD = time.perf_counter() - start_time_update
                    self.update_elapsed_time.append(executionTimeUPD)

                    self.benchmark_tool.set_value(
                        'update_clock',
                        time.perf_counter()
                    )
                    self.benchmark_tool.set_value(
                        'update_time',
                         executionTimeUPD
                         )


                    startTimeREADPRO = time.perf_counter()


                    total = 0
                    signal = bytearray()
                    total_size = self.na_read*self.n_ax*self.n_el*self.bytesPerElementRead
                    # Ensure that the complete data is received
                    while total < total_size:
                        # For best match with hardware and network realities, the value of bufsize
                        # should be a relatively small power of 2, for example, 4096.
                        data = connection.recv(self.buffer_size)
                        if not data:
                            break
                        total = total + len(data)
                        signal += data

                    received_serial_data = np.frombuffer(
                        signal, dtype=np.int16)

                    self.reading_clock.append(time.perf_counter())

                    RF = received_serial_data.reshape(
                        self.n_ax, self.na_read, self.n_el, order='F')
                    RF = np.transpose(RF, (1, 2, 0))
                    IQ = np.empty(
                        (1, self.na_read, self.n_el, self.n_ax//2, 2))

                    # Extract I and Q componenets from RF
                    IQ[:, :, :, :, 0] = RF[:, :, 1::2]
                    IQ[:, :, :, :, 1] = RF[:, :, ::2]
                    buf = {'data': IQ, 'frame_id': self.global_id}
                    self.global_id += 1
                    buffer.append(buf)

                    executionTimeREADPRO = time.perf_counter() - startTimeREADPRO
                    self.read_preprocess_elapsed_time.append(
                        executionTimeREADPRO)

                    self.benchmark_tool.set_value(
                        'read_clock', time.perf_counter()
                    )
                    self.benchmark_tool.set_value(
                        'read_time', executionTimeREADPRO
                        )


            except:
                logging.debug('set terminate signal...')
                terminate_signal.set()  # signal writer that we are done
                print('TCP server closes...')
                connection.close()
                connection = None
                buffer.clear()

        #time.sleep(0.0001)  # sleep for 1 ms
        terminate_signal.set()
        buffer.clear()  # clear buffer if while loop is terminated

    def save(self):
        """Function that saves benchmark data to .mat file"""
        if self.saving:
            # difference of clock list
            processingInterFramePeriod = np.diff(self.processing_clock)
            displayInterFramePeriod = np.diff(self.display_clock)
            readingInterFramePeriod = np.diff(self.reading_clock)

            d = {
                f"displayIFP_{int(self.na_transmit)}": displayInterFramePeriod,
                f"readingIFP_{int(self.na_transmit)}": readingInterFramePeriod,
                f"processingIFP_{int(self.na_transmit)}": processingInterFramePeriod,
                f"tBeamformerElapsedTime_{int(self.na_transmit)}": self.beamformer_elapsed_time,
                f"tUpdateElapsedTime_{int(self.na_transmit)}": self.update_elapsed_time,
                f"tReadPreProcessElapsedTime_{int(self.na_transmit)}":
                self.read_preprocess_elapsed_time,
                f"diplayClock_{int(self.na_transmit)}": self.display_clock,
                f"processingClock_{int(self.na_transmit)}": self.processing_clock,
                f"readingClock_{int(self.na_transmit)}": self.reading_clock,
                f"processing_id_{int(self.na_transmit)}": self.processing_id,
            }

            filename = (
                f"L114v_na{int(self.na_transmit)}"
                f"_proc{self.bf_type}"
                f"_autotun{str(self.auto_update_intensity)}"
                f"_pyt_{int(time.time())}.mat"
            )

            scipy.io.savemat(filename, mdict=d)

            # clear lists
            self.beamformer_elapsed_time = []
            self.read_preprocess_elapsed_time = []
            self.update_elapsed_time = []
            self.processing_clock = []
            self.display_clock = []
            self.reading_clock = []
            self.processing_id = []
            self.global_id = 0

    def process_data(self, buffer, terminate_signal):
        """Function that handles data processing (e.g. beamforming)"""
        while not terminate_signal.is_set():
            logging.debug('start processing')
            time.sleep(0.)
            try:
                startTimeBF = time.perf_counter()
                inputs, frame_id = self.prepare_inputs(buffer)

                if inputs:
                    BF = self.active_model(inputs)[0]
                    img = self.scan_converter.convert(BF)
                    img = np.array(img)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # FPS counter
                    if self.show_fps:
                        img = self.fps_counter.overlay(img)

                    if self.benchmark_tool.is_running:
                        cv2.putText(img,
                                    f'benchmark active: {self.benchmark_tool.current_benchmark}',
                                    (7, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 0, 0, 255),
                                    1,
                                    cv2.LINE_AA
                                    )

                    self.bf_display = img

                    # Saving parameters
                    self.processing_clock.append(time.perf_counter())
                    executionTimeBF = time.perf_counter() - startTimeBF
                    self.beamformer_elapsed_time.append(executionTimeBF)
                    self.processing_id.append(frame_id)

                    self.benchmark_tool.set_value(
                        'processing_id',
                        frame_id
                    )
                    self.benchmark_tool.set_value(
                        'processing_clock',
                        time.perf_counter()
                    )
                    self.benchmark_tool.set_value(
                        'processing_time',
                        executionTimeBF
                    )
            except:
                buffer.clear()
                #time.sleep(0.0001)  # wait for new data

        #buffer.clear()

    def open_connection(self):
        """Handles the opening and handshaking with the Verasonics"""
        # Open connection
        connection, client_address = self.vera_socket.accept()
        print('Connected by', client_address)
        # READ INITIALIZATION PARAMETERS
        total = 0
        msg = bytearray()

        while total < 7:  # Ensure to receive the complete data
            # For best match with hardware and network realities,
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

        matchcase = initializationParameters[7]

        if matchcase == 1:
            self.bf_type = 'RAW'
        elif matchcase == 2:
            self.bf_type = 'DAS'
        elif matchcase == 3:
            self.bf_type = 'ABLE'
        else:
            logging.error('Beamformer type does not match.')


        # ACK INITIALIZATION COMPLETED
        ack = [1]
        ack_bytes = bytearray(struct.pack(f'{len(ack)}B', *ack))
        connection.sendall(ack_bytes)

        logging.debug('Initialization completed')
        return connection

    def get_frame(self):
        """Function that generates new image frames for the web interface"""
        while True:
            try:
                encoded = self.encode_img(self.bf_display)
                self.bf_display = None

                self.display_clock.append(time.perf_counter())

                self.benchmark_tool.set_value(
                    'display_clock',
                    time.perf_counter()
                    )

                yield encoded
            except:
                pass
                #time.sleep(0.0001)  # wait for new data

    @staticmethod
    def encode_img(img):
        """Function that encodes the input image to jpeg and prepares the HTML package"""
        imgencode = cv2.imencode('.jpg', img)[1]
        stringData = imgencode.tobytes()
        return (
            b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n'
        )

    def select_model(self, bf_type, na_transmit):
        """Function that selects the correct model for the beamformer"""

        if self.bf_type == 'RAW':
            def return_IQ(inputs):
                IQ = inputs['data']
                ix = int(np.floor(IQ.shape[1]/2))
                return tf.transpose(IQ[:, ix, :, :, :], perm=(0, 2, 1, 3))
            model = return_IQ
        if bf_type == 'DAS':
            key = 'DAS_' + str(na_transmit) + 'PW'
            model = self.model_dict[key]
        elif bf_type == 'ABLE':
            key = 'ABLE_' + str(na_transmit) + 'PW'
            model = self.model_dict[key]

        return model

    def update_settings(self):
        """Function that updates setting states of the webserver class"""

        # Hacky fix, in the future lets handle all requests as json
        # pylint: disable=assigning-non-slot
        if request.is_json:
            json_data = request.get_json()
            request.form = json_data # convert json string to dict

        sent_from = request.form.get('sent_from')


        if (request.form.get('beamformer') is not None) or (request.form.get('na') is not None):
            self.save()

        print("Updating settings:", str(request.form))

        # Do not update settings from GUI while benchmark is running
        if (sent_from != 'benchmark_tool') and self.benchmark_tool.is_running:
            print("WARNING: Benchmark is running. Do not update settings!")
            return ('', 204)

        self.terminate_signal.set()  # Terminate data processing while updating settings

        if request.method == 'POST':
            if request.form.get('beamformer') is not None:

                self.update_beamformer = True
                self.bf_type = request.form.get('beamformer')
                # Update model
                self.active_model = self.select_model(
                    self.bf_type, self.na_transmit)
                logging.debug(self.bf_type)

            if request.form.get('norm_mode') is not None:
                norm_mode = request.form.get('norm_mode')
                setattr(self.scan_converter, 'norm_mode', norm_mode)
                logging.debug(norm_mode)

            if request.form.get('na') is not None:

                self.update_na_transmit = True
                self.na_transmit = int(request.form.get('na'))
                # Update model
                self.active_model = self.select_model(
                    self.bf_type, self.na_transmit)
                logging.debug(self.na_transmit)

            if request.form.get('intensityAuto') is not None:
                self.auto_update_intensity = not self.auto_update_intensity
                logging.debug(self.auto_update_intensity)
                if self.auto_update_intensity:
                    self.meanAmp_history = []
                    self.err_history = [0, 0]
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
                self.scan_converter.alpha = float(
                    request.form.get('slide_alpha'))
                logging.debug(self.scan_converter.alpha)

            if request.form.get('slide_sos') is not None:
                c_prev = self.c
                self.c = float(request.form.get('slide_sos'))

                # rescale grid
                self.grid[:, :, 2] = self.grid[:, :, 2] * self.c/c_prev
                logging.debug(self.c)

            if request.form.get('slide_fs') is not None:
                self.fs = float(request.form.get('slide_fs'))*1e6
                logging.debug(self.fs)

            if request.form.get('slide_fc') is not None:
                self.fc = float(request.form.get('slide_fc'))*1e6
                logging.debug(self.fc)

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

                # clear lists
                self.beamformer_elapsed_time = []
                self.read_preprocess_elapsed_time = []
                self.update_elapsed_time = []
                self.processing_clock = []
                self.display_clock = []
                self.reading_clock = []
                self.processing_id = []
                self.global_id = 0

                logging.debug(self.source)

            if request.form.get('benchmark') is not None:
                if request.form.get('benchmark'):
                    self.benchmark_tool.run()

        return ('', 204)

    def get_settings(self):
        """Function that returns the current settings of the webserver class"""
        import json
        def is_jsonable(x):
            try:
                json.dumps(x)
                return True
            except (TypeError, OverflowError):
                return False

        settings = self.__dict__
        for key,value in settings.items():
            if not is_jsonable(value):
                settings[key] = str(value)

        return settings

    def delete_settings(self):
        """Function that resets the current settings of the webserver class"""
        self.__init__()

    def set_settings(self, settings):
        """Function that sets the current settings of the webserver class"""
        self.__dict__ = settings


# Initialize Flask server
app = Flask(__name__)
app.secret_key = 'Secret'
app.config["SESSION_PERMANENT"] = False

@app.route('/')
def index():
    """Creates the main HTML page"""
    return render_template('index.html')


@app.route('/video')
def video():
    """Creates the video HTML page"""
    return render_template('video.html')


@app.route('/create_file', methods=['POST'])
def create_file():
    """Function that updates server settings based on user input"""
    return usp.update_settings()


@app.route('/vid/')
def vid():
    """ Video Feed"""
    return Response(usp.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_benchmark', methods=['POST'])
def start_benchmark():
    """Function that starts the benchmark tool"""
    return usp.benchmark_tool.run()

# Initialize Verasonics webserver
usp = UltrasoundProcessingServer()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,
            debug=debugger_is_active(), use_reloader=False)
