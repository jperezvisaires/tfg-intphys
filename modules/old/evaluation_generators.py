import numpy as np
import h5py
from skimage.io import imread
from skimage.transform import rescale
from tensorflow.keras.utils import Sequence, to_categorical

class unet_seg2seg_generator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 model_object,
                 model_occlu,
                 input_frames=4,
                 batch_size=1, 
                 dim=(288, 288), 
                 scale=0.5, 
                 num_channels=2,  
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.input_frames = input_frames
        self.path_hdf5 = path_hdf5
        self.model_object = model_object
        self.model_occlu = model_occlu
        self.scale = scale
        self.dim_scaled = (int(self.dim[0] * self.scale), int(self.dim[1] * self.scale))
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        '''Updates indexes after each epoch'''
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __rescale(self, array):
        
        if self.scale != 1:
            array = rescale(image=array, 
                            scale=self.scale, 
                            order=1, 
                            preserve_range=True, 
                            multichannel=True, 
                            anti_aliasing=False)
        
        return array

    def __data_generation(self, list_samples_temp):
        '''Generates data containing batch_size samples'''
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim_scaled, self.num_channels * self.input_frames))
        Y = np.empty((self.batch_size, *self.dim_scaled, 1))       

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample               
                array_list = []
                
                for j in range(self.input_frames):

                    image = self.__rescale(f[sample[j]][:])
                    image = np.reshape(image, newshape=(1, image.shape[0], image.shape[1], 3))
                    array_list.append(self.model_object(image))
                    array_list.append(self.model_occlu(image))

                X[i,] = np.concatenate(array_list, axis=-1)

                # Store target
                target = self.targets[sample[0]]

                image = self.__rescale(f[target][:])
                image = np.reshape(image, newshape=(1, image.shape[0], image.shape[1], 3))

                Y[i,] = (self.model_object(image))


        return X, Y