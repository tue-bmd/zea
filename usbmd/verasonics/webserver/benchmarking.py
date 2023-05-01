import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import threading
import requests
import json

from usbmd.utils.config import load_config_from_yaml


# Use this benchmark as basis for pytest

class BenchmarkTool:
    """ Class that handles benchmarking of the cloud based ultrasound system"""

    def __init__(self, output_folder, benchmark_config):
        # Create unique folder for this benchmark
        self.output_folder = os.path.join(
            output_folder, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.output_folder, exist_ok=True)
        self.data = pd.DataFrame(
            columns=['processing_id',
                     'processing_time',#
                     'read_time', #
                     'update_time', #
                     'processing_clock',#
                     'read_clock', #
                     'update_clock',#
                     'display_clock'#
                     ]
        )
        self.is_running = False
        self.current_benchmark = None

        self.data_buffer = []

        # Load benchmark config
        self.config = load_config_from_yaml(benchmark_config)

        print('Benchmark tool initialized')

    def set_value(self, column, value):
        """append a value to the dataframe in the specified column"""
        #self.data.loc[len(self.data), column] = value
        self.data_buffer.append([column, value])

    def purge_to_dataframe(self):
        """Append the data in the buffer to the dataframe"""
        for column, value in self.data_buffer:
            self.data.loc[len(self.data), column] = value

        self.data_buffer = []

    def run(self):
        """Starts the benchmarking process"""
        benchmark_thread = threading.Thread(target=self.benchmark)
        benchmark_thread.daemon = True
        benchmark_thread.start()
        return

    def benchmark(self):
        """Runs the benchmark"""
        self.is_running = True
        """Runs a single benchmark"""
        print('Starting benchmark')

        for name, params in self.config.items():
            # Let the server know this request is sent from the benchmark tool
            params['sent_from'] = 'benchmark_tool'

            # Update server settings
            response = requests.post('http://localhost:5000/create_file', json=params)

            if response.status_code == 204:
                self.current_benchmark = name
                self.clear()
                # Wait for the specified amount of time
                time.sleep(params['duration'])

                # Save the benchmark data
                self.current_benchmark = None
                self.save(name, format='xlsx')
                self.clear()



        self.is_running = False
        print('Benchmark finished')

    def save(self, name, format='csv'):
        """Saves the benchmark data to a file"""

        self.purge_to_dataframe()
        snapshot = self.data.copy()

        savepath = os.path.join(self.output_folder, name)

        if format == 'csv':
            snapshot.to_csv(savepath+'.csv')
        elif format == 'xlsx':
            snapshot.to_excel(savepath+'.xlsx')
        elif format == 'mat':
            raise NotImplementedError('Saving to .mat not yet implemented')
        else:
            raise ValueError('format must be csv, xlsx or mat')

        print(f'Saved benchmark data to {savepath}.{format}')


    def clear(self):
        """Clears the benchmark data"""
        self.data_buffer = []
        self.data = pd.DataFrame(columns=self.data.columns)
