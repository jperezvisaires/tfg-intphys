import numpy as np
import h5py
from skimage.io import imread
from skimage.transform import rescale
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical

class SegmentationObjectDataGenerator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 batch_size=32, 
                 dim=(288, 288), 
                 scale=1, 
                 num_channels=3, 
                 num_classes=1, 
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.path_hdf5 = path_hdf5
        self.scale = scale
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __int_to_float(self, array, bits):
    
        array = array.astype(np.float32) / (2**bits - 1)
    
        return array

    def __mask_oneshot(self, mask_array):
    
        oneshot_array = np.empty((mask_array.shape))
        oneshot_array[mask_array == 3] = 1
        oneshot_array[mask_array != 3] = 0
    
        return oneshot_array
    
    def __rescale(self, array):
        
        if self.scale != 1:
            array = rescale(image=array, 
                            scale=self.scale, 
                            order=0, 
                            preserve_range=True, 
                            multichannel=True, 
                            anti_aliasing=False)
        
        return array

    def __data_generation(self, list_samples_temp):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample
                X[i,] = f[sample][:]

                # Store target
                target = self.targets[sample]
                Y[i,] = f[target][:]
        
        X = self.__int_to_float(X, 8)
        Y = self.__mask_oneshot(Y)
        X = self.__rescale(X)
        Y = self.__rescale(Y)
        
        return X, Y

class SegmentationOccluderDataGenerator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 batch_size=32, 
                 dim=(288, 288), 
                 scale=1, 
                 num_channels=3, 
                 num_classes=1, 
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.path_hdf5 = path_hdf5
        self.scale = scale
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __int_to_float(self, array, bits):
    
        array = array.astype(np.float32) / (2**bits - 1)
    
        return array
    
    def __mask_oneshot(self, mask_array):
    
        oneshot_array = np.empty((mask_array.shape))
        oneshot_array[mask_array == 4] = 1
        oneshot_array[mask_array != 4] = 0
    
        return oneshot_array
    
    def __rescale(self, array):
        
        if self.scale != 1:
            array = rescale(image=array, 
                            scale=self.scale, 
                            order=0, 
                            preserve_range=True, 
                            multichannel=True, 
                            anti_aliasing=False)
        
        return array

    def __data_generation(self, list_samples_temp):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample
                X[i,] = f[sample][:]

                # Store target
                target = self.targets[sample]
                Y[i,] = f[target][:]
        
        X = self.__int_to_float(X, 8)
        Y = self.__mask_oneshot(Y)
        X = self.__rescale(X)
        Y = self.__rescale(Y)
        
        return X, Y

class DepthDataGenerator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 batch_size=32, 
                 dim=(288, 288), 
                 scale=1, 
                 num_channels=3,  
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.path_hdf5 = path_hdf5
        self.scale = scale
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __int_to_float(self, array, bits):
    
        array = array.astype(np.float32) / (2**bits - 1)
    
        return array
    
    def __rescale(self, array):
        
        if self.scale != 1:
            array = rescale(image=array, 
                            scale=self.scale, 
                            order=0, 
                            preserve_range=True, 
                            multichannel=True, 
                            anti_aliasing=False)
        
        return array

    def __data_generation(self, list_samples_temp):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample
                X[i,] = f[sample][:]

                # Store target
                target = self.targets[sample]
                Y[i,] = f[target][:]
        
        X = self.__int_to_float(X, 8)
        Y = self.__int_to_float(Y, 16)
        X = self.__rescale(X)
        Y = self.__rescale(Y)
        
        return X, Y

class PredictionConvLSTMDataGenerator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 input_frames=3,
                 batch_size=32, 
                 dim=(288, 288), 
                 scale=1, 
                 num_channels=3,  
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.input_frames = input_frames
        self.path_hdf5 = path_hdf5
        self.scale = scale
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __int_to_float(self, array, bits):
    
        array = array.astype(np.float32) / (2**bits - 1)
    
        return array
    
    def __mask_oneshot_object(self, mask_array):
    
        oneshot_array = np.empty((mask_array.shape))
        oneshot_array[mask_array == 3] = 1
        oneshot_array[mask_array != 3] = 0.5
    
        return oneshot_array

    def __mask_oneshot_occluder(self, mask_array):
    
        oneshot_array = np.empty((mask_array.shape))
        oneshot_array[mask_array == 4] = 1
        oneshot_array[mask_array != 4] = 0
    
        return oneshot_array
    
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
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, self.input_frames, *(144, 144), self.num_channels))
        Y = np.empty((self.batch_size, *(144, 144), self.num_channels))
        Z = np.empty((self.batch_size, *(144, 144), self.num_channels * self.input_frames))

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample               
                array_list = []
                
                for j in range(self.input_frames):
                    array_list.append(self.__rescale((f[sample[j]][:]/4)))

                Z[i,] = np.concatenate(array_list, axis=-1)
                X[i,] = np.reshape(Z[i,], (self.input_frames, *(144, 144), self.num_channels))

                # Store target
                target = self.targets[sample[0]]
                Y[i,] = self.__rescale(f[target][:]/4)

        return X, Y

class PredictionShortDataGenerator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 memory_frames=3,
                 batch_size=32, 
                 dim=(288, 288), 
                 scale=1, 
                 num_channels=3,  
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.memory_frames = memory_frames
        self.path_hdf5 = path_hdf5
        self.scale = scale
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __int_to_float(self, array, bits):
    
        array = array.astype(np.float32) / (2**bits - 1)
    
        return array
    
    def __rescale(self, array):
        
        if self.scale != 1:
            array = rescale(image=array, 
                            scale=self.scale, 
                            order=1, 
                            preserve_range=True, 
                            multichannel=True, 
                            anti_aliasing=True)
        
        return array

    def __data_generation(self, list_samples_temp):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels * self.memory_frames))
        Y = np.empty((self.batch_size, *self.dim, self.num_channels))

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample               
                array_list = []
                
                for j in range(self.memory_frames):
                    array_list.append(f[sample[j]][:])

                X[i,] = np.concatenate(array_list, axis=-1)

                # Store target
                target = self.targets[sample[0]]
                Y[i,] = f[target][:]
        
        X = self.__int_to_float(X, 8)
        Y = self.__int_to_float(Y, 8)
        X = self.__rescale(X)
        Y = self.__rescale(Y)
        
        return X, Y

class PredictionLongDataGenerator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 memory_frames=5,
                 batch_size=32, 
                 dim=(288, 288), 
                 scale=1, 
                 num_channels=3,  
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.memory_frames = memory_frames
        self.path_hdf5 = path_hdf5
        self.scale = scale
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __int_to_float(self, array, bits):
    
        array = array.astype(np.float32) / (2**bits - 1)
    
        return array
    
    def __rescale(self, array):
        
        if self.scale != 1:
            array = rescale(image=array, 
                            scale=self.scale, 
                            order=1, 
                            preserve_range=True, 
                            multichannel=True, 
                            anti_aliasing=True)
        
        return array

    def __data_generation(self, list_samples_temp):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels * self.memory_frames))
        Y = np.empty((self.batch_size, *self.dim, self.num_channels))

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample               
                array_list = []
                
                for j in range(self.memory_frames):
                    array_list.append(f[sample[j]][:])

                X[i,] = np.concatenate(array_list, axis=-1)

                # Store target
                target = self.targets[sample[0]]
                Y[i,] = f[target][:]
        
        X = self.__int_to_float(X, 8)
        Y = self.__int_to_float(Y, 8)
        X = self.__rescale(X)
        Y = self.__rescale(Y)
        
        return X, Y

class PredictionOpticDataGenerator(Sequence):
    'Generates data for Keras fit training function'

    def __init__(self, 
                 list_samples, 
                 targets, 
                 path_hdf5,
                 memory_frames=5,
                 batch_size=10, 
                 dim=(288, 288), 
                 scale=1, 
                 num_channels=3,  
                 shuffle=True,
                 first=True):
        'Initialization'
        
        self.dim = dim
        self.memory_frames = memory_frames
        self.path_hdf5 = path_hdf5
        self.scale = scale
        self.batch_size = batch_size
        self.targets = targets
        self.list_samples = list_samples
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.first = first
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return int(np.floor(len(self.list_samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of samples
        list_samples_temp = [self.list_samples[i] for i in indexes]

        # Generate data
        X, Y = self.__data_generation(list_samples_temp)

        return X, Y

    def on_epoch_end(self):        
        'Updates indexes after each epoch'
        
        self.indexes = np.arange(len(self.list_samples))
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __int_to_float(self, array, bits):
    
        array = array.astype(np.float32) / (2**bits - 1)
    
        return array
    
    def __rescale(self, array):
        
        if self.scale != 1:
            array = rescale(image=array, 
                            scale=self.scale, 
                            order=1, 
                            preserve_range=True, 
                            multichannel=True, 
                            anti_aliasing=True)
        
        return array

    def __data_generation(self, list_samples_temp):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.num_channels * self.memory_frames))
        Y = np.empty((self.batch_size, *self.dim, self.num_channels + self.num_channels * self.memory_frames))

        # Generate data            
        for i, sample in enumerate(list_samples_temp):
            
            with h5py.File(self.path_hdf5, "r") as f:    
                
                # Store sample               
                array_list = []
                
                for j in range(self.memory_frames):
                    array_list.append(f[sample[j]][:])

                X[i,] = np.concatenate(array_list, axis=-1)

                # Store target
                target = self.targets[sample[0]]
                array_list.append(f[target][:])
                Y[i,] = np.concatenate(array_list, axis=-1)

        X = self.__int_to_float(X, 8)
        Y = self.__int_to_float(Y, 8)
        X = self.__rescale(X)
        Y = self.__rescale(Y)
        
        return X, Y