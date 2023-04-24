import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import threading
import requests


class BenchmarkTool:
    """ Class that handles benchmarking of the cloud based ultrasound system"""
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.data = pd.DataFrame(
            columns=['id',
                     'processing_time',
                     'read_time',
                     'processing_IFP'
                     'read_IFP',
                     'update_IFP',
                     'display_IFP'
            ]
        )
        self.is_running = False


    def set_value(self, column, value):
        """append a value to the dataframe in the specified column"""
        self.data.loc[len(self.data), column] = value

    def run(self):
        """Starts the benchmarking process"""
        benchmark_thread = threading.Thread(target=self.benchmark)
        benchmark_thread.daemon = True
        benchmark_thread.start()

        benchmark_thread.wait()
        return



    def benchmark(self):
        self.is_running = True
        """Runs a single benchmark"""
        print('Starting benchmark')

        time.sleep(1)
        print('1 second sleep done')
        time.sleep(1)
        print('2 second sleep done')
        time.sleep(1)
        print('3 second sleep done')
        time.sleep(1)
        print('4 second sleep done')
        time.sleep(1)

        self.is_running = False
        print('Benchmark finished')

    def save(self, format='csv'):
        """Saves the benchmark data to a file"""

        savepath = os.path.join(self.output_folder, datetime.now().strftime('%Y%m%d_%H%M%S'))

        if format == 'csv':
            self.data.to_csv(os.path.join(savepath, 'benchmark.csv'))
        elif format == 'xlsx':
            self.data.to_excel(os.path.join(savepath, 'benchmark.xlsx'))
        elif format == 'mat':
            raise NotImplementedError('Saving to .mat not yet implemented')
        else:
            raise ValueError('format must be csv, xlsx or mat')
