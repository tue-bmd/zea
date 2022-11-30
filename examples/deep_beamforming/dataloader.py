from glob import glob
import random
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert

from sklearn.model_selection import train_test_split
import cv2

eps = 1e-10

def get_dataloader(cfg, probe, grid):

    if cfg.data.dataset == 'US_Data':
        return Dataloader_L114v(cfg, probe, grid)

    if cfg.data.dataset == 'LifeTec2022_test':
        return Dataloader_S51(cfg, probe, grid)


class Dataloader():
    
    def __init__(self, cfg, probe, grid):
        self.cfg = cfg
        self.probe = probe
        self.grid = grid
        self.data_root = cfg.data.data_path
        self.dataset = cfg.data.dataset
        self.target_type = cfg.data.target_type
        self.input_type = cfg.data.input_type

        self.input_paths = sorted(glob('%s/%s/%s/%s/*' % (self.data_root, self.dataset, 'train', cfg.data.input_type)))
        self.target_paths = sorted(glob('%s/%s/%s/%s/*' % (self.data_root, self.dataset, 'train', cfg.data.target_type)))
        
        self.input_paths_test = sorted(glob('%s/%s/%s/%s/*' % (self.data_root, self.dataset, 'test', cfg.data.input_type)))
        self.target_paths_test = sorted(glob('%s/%s/%s/%s/*' % (self.data_root, self.dataset, 'test', cfg.data.target_type)))
        
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.input_paths, 
                                                                              self.target_paths, 
                                                                              test_size=0.1, 
                                                                              random_state=1)
        
        print('Found %d train samples, %d validation samples, %d test samples.' % (len(self.x_train),len(self.x_val), len(self.target_paths_test))) 

        self.demodulate = cfg.data.get('IQ')
        self.angles = getattr(cfg.data, 'angles')
        self.env_detect_target = cfg.data.get('env_detect_targets')
        self.patch_shape = cfg.model.get('patch_shape') #getattr(cfg, 'patch_shape', None) #If not patch size is given, use full gridgrid.shape[:2]
        self.Nx = grid.shape[1]
        self.Nz = grid.shape[0]
        
    def load_batches(self, validating=False, store_in_memory=True):
        
        if validating:
            combined_paths = list(zip(self.x_val, self.y_val))
        else:
            combined_paths = list(zip(self.x_train, self.y_train))
            
        random.shuffle(combined_paths)
        
        inputs_buffer = []
        targets_buffer = []
        
        if not validating:
            aug = lambda x: x
            # Currently no augmentation implemented. Can be added here.
        else:
            aug = lambda x: x
        
        while True:
            for input_path, target_path in combined_paths:
                
                input_data, target_data = self.process_mat(input_path, target_path)
                                
                if store_in_memory:
                    inputs_buffer.append(input_data)
                    targets_buffer.append(target_data)

                # Provide input_data and target_data to the augmentation function simultaneously to ensure the same augmentation behavior for each x-y pair.
                input_data, target_data = aug([input_data, target_data])
                
                #Extract patch
                target_data, grid = self.get_patch(target_data)
                
                yield ((input_data, grid), target_data)
                    
            while store_in_memory:
                for input_data, target_data in zip(inputs_buffer,targets_buffer):
                    input_data, target_data = aug([input_data, target_data])
                       
                    #Extract patch
                    target_data, grid = self.get_patch(target_data)
                    
                    yield ((input_data, grid), target_data)


    def load_test(self):
        input_paths = self.input_paths_test
        target_paths = self.target_paths_test
        
        test_X, test_Y = [], []
        
        for input_path, target_path in zip(input_paths, target_paths):
                
            input_data, target_data = self.process_mat(input_path, target_path)
            
            test_X.append(input_data)
            test_Y.append(target_data)
            
        return test_X, test_Y
        

    def process_mat(self, input_path, target_path):
        
        #Load input data
        matfile = sio.loadmat(input_path)
        input_data = np.array(matfile['rf_data'], dtype='float32')
        input_data = np.transpose(input_data,(2,1,0))
        input_data = np.expand_dims(input_data,axis=3)
        input_data = np.nan_to_num(input_data)
        
        #Convert RF signals to IQ
        if self.demodulate:
            idata = np.real(input_data)
            qdata = np.imag(hilbert(idata, axis=2))
            input_data = np.concatenate((idata,qdata),axis=3)

            #input_data = input_data[:,:,::4,:] ################################## REMOVE!!!!

        
        #Select PW inputs        

        if self.dataset == 'MRI2US': # If simulated MRI to US data
            input_data = input_data/input_data.max(axis=(1,2,3), keepdims=True)

            target_data = matfile['im']
            target_data = cv2.resize(target_data.T, (self.Nx, self.Nz))

            return input_data[self.angles,:,:,:], target_data

        else: #if normal US data

            #messy fix, better to change .mat files later on
            if self.target_type == 'output_11pw_mv_preenvelope':
                key = 'mv_pre_envelope_11pw'
            elif self.target_type == 'output_1pw_mv_preenvelope':
                key = 'mv_pre_envelope_1pw'

            #Load target data
            matfile = sio.loadmat(target_path)           
            target_data = np.array(matfile[key], dtype='float32')
            target_data = np.squeeze(target_data)
    
            if self.env_detect_target:
                target_data = np.abs(hilbert_tf(target_data))

            # reshape if needed
            target_shape = target_data.shape
            network_shape = self.grid.shape[:2]

            if not target_shape == network_shape:
                target_data = cv2.resize(target_data, dsize=network_shape[::-1])

             
            return input_data[self.angles,:,:,:]/2**15, target_data/2**15
        
    
    def get_patch(self, target):
        
        if self.patch_shape:
            if self.patch_shape[0] == self.Nx:
                x_ix = 0
            else:
                x_ix = np.random.randint(0,self.Nx-self.patch_shape[0])
                
            if self.patch_shape[1] == self.Nz:
                z_ix = 0
            else:
                z_ix = np.random.randint(0,self.Nz-self.patch_shape[1])
        else:
            x_ix = 0
            z_ix = 0
          
        grid_patch = self.grid[z_ix:z_ix+self.patch_shape[1], x_ix:x_ix+self.patch_shape[0], :]
        grid_patch = np.transpose(grid_patch, axes=(1,0,2))
        
        target_patch = target[z_ix:z_ix+self.patch_shape[1],x_ix:x_ix+self.patch_shape[0]]
        
        return target_patch, grid_patch




class Dataloader_L114v(Dataloader):
    
    def __init__(self, cfg, probe, grid):
        super().__init__(cfg, probe, grid)

    

class Dataloader_S51(Dataloader):

    def __init__(self, cfg, probe, grid):
        super().__init__(cfg, probe, grid)


    def process_mat(self, input_path, target_path):
        
        #Load input data
        matfile = sio.loadmat(input_path)
        input_data = np.array(matfile['rf_data'], dtype='float32')
        input_data = np.transpose(input_data,(3,1,2,0))
        input_data = np.expand_dims(input_data,axis=-1)
        input_data = np.nan_to_num(input_data)
        
        #Convert RF signals to IQ
        if self.demodulate:
            idata = np.real(input_data)
            qdata = np.imag(hilbert(idata, axis=-2))
            input_data = np.concatenate((idata,qdata),axis=-1)


        input_data = input_data[0,self.angles,:,:,:]/2**15

        #Load target data, just random data atm for testing purposes
        target_data = np.random.rand(self.Nz, self.Nx)
            
        return input_data, target_data
            


    





