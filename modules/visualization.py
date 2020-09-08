import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.io import imread
from skimage.transform import rescale
from tensorflow.keras.utils import to_categorical

def int_to_float(array, bits):
    
    array = array.astype(np.float32) / (2**bits - 1)
    
    return array

def mask_oneshot_object(mask_array):
    
    oneshot_array = np.empty((mask_array.shape))
    oneshot_array[mask_array == 3] = 1.0
    oneshot_array[mask_array != 3] = 0.0
    
    return oneshot_array

def mask_oneshot_occlu(mask_array):
    
    oneshot_array = np.empty((mask_array.shape))
    oneshot_array[mask_array == 4] = 1.0
    oneshot_array[mask_array != 4] = 0.0
    
    return oneshot_array

def rescale_array(array, scale):
        
    if scale != 1:

        array = rescale(image=array, 
                        scale=scale, 
                        order=1, 
                        preserve_range=True, 
                        multichannel=True, 
                        anti_aliasing=False)
        
    return array

def prediction_mask_object(path_hdf5, model, scene=1, frame=1, scale=0.5):

    with h5py.File(path_hdf5, "r") as f:
        
        image = f['train/{:0>5d}/scene/scene_{:0>3d}'.format(scene, frame)][:]
        mask = f['train/{:0>5d}/masks/masks_{:0>3d}'.format(scene, frame)][:]

    plt.figure(figsize = (10, 30))
    
    ax = plt.subplot(1, 3, 1)
    image = rescale_array(image, scale)
    image = int_to_float(image, 8)
    plt.imshow(image)

    ax = plt.subplot(1, 3, 2)
    mask = mask_oneshot_object(mask)
    mask = rescale_array(mask, scale)
    plt.imshow(np.reshape(mask, (mask.shape[0], mask.shape[1])), cmap="viridis", vmin=0, vmax=1)

    ax = plt.subplot(1, 3, 3)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    prediction = model(image)
    plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2])), cmap="viridis", vmin=0, vmax=1)
    
    plt.show()

def prediction_mask_occlu(path_hdf5, model, scene=1, frame=1, scale=0.5):

    with h5py.File(path_hdf5, "r") as f:
        image = f['train/{:0>5d}/scene/scene_{:0>3d}'.format(scene, frame)][:]
        mask = f['train/{:0>5d}/masks/masks_{:0>3d}'.format(scene, frame)][:]

    plt.figure(figsize = (10, 30))
    
    ax = plt.subplot(1, 3, 1)
    image = int_to_float(image, 8)
    image = rescale_array(image, scale)
    plt.imshow(image)

    ax = plt.subplot(1, 3, 2)
    mask = rescale_array(mask, scale)
    mask = mask_oneshot_occlu(mask)
    plt.imshow(np.reshape(mask, (mask.shape[0], mask.shape[1])), cmap="viridis", vmin=0, vmax=1)

    ax = plt.subplot(1, 3, 3)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    prediction = model(image)
    plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2])), cmap="viridis", vmin=0, vmax=1)
    
    plt.show()

def prediction_mask_10(path_hdf5, model, scene=1, frame=50, scale=1):

    plt.figure(figsize = (100, 300))

    for i in range(10): 
    
        with h5py.File(path_hdf5, "r") as f:
            image = f['train/{:0>5d}/scene/scene_{:0>3d}'.format(scene, frame + i)][:]
            mask = f['train/{:0>5d}/masks/masks_{:0>3d}'.format(scene, frame + i)][:]
    
        ax = plt.subplot(10, 3, 1 + 3 * i)
        image = int_to_float(image, 8)
        image = rescale(image=image, scale=scale, multichannel=True, order=0)
        plt.imshow(image)

        ax = plt.subplot(10, 3, 2 + 3 * i)
        mask = rescale(image=mask, scale=scale, multichannel=True, order=0, preserve_range=True, anti_aliasing=False)
        plt.imshow(np.reshape(mask, (mask.shape[0], mask.shape[1])), cmap="viridis", vmin=0, vmax=4)

        ax = plt.subplot(10, 3, 3 + 3 * i)
        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        prediction = model.predict(x=image)
        plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2])), cmap="viridis", vmin=0, vmax=1)
    
    plt.show()

def prediction_depth(path_hdf5, model, scene=1, frame=50, scale=1):

    with h5py.File(path_hdf5, "r") as f:
        image = f['train/{:0>5d}/scene/scene_{:0>3d}'.format(scene, frame)][:]
        depth = f['train/{:0>5d}/depth/depth_{:0>3d}'.format(scene, frame)][:]

    plt.figure(figsize = (10, 30))
    
    ax = plt.subplot(1, 3, 1)
    image = int_to_float(image, 8)
    image = rescale(image=image, scale=scale, multichannel=True, order=0)
    plt.imshow(image)

    ax = plt.subplot(1, 3, 2)
    depth = int_to_float(depth, 16)
    depth = rescale(image=depth, scale=scale, multichannel=True, order=0, preserve_range=True, anti_aliasing=False)
    plt.imshow(np.reshape(depth, (depth.shape[0], depth.shape[1])), cmap="viridis", vmin=0, vmax=1)

    ax = plt.subplot(1, 3, 3)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    prediction = model.predict(x=image)
    plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2])), cmap="viridis", vmin=0, vmax=1)
    
    plt.show()

def convlstm_depth(path_hdf5, model, scene=1, frame=50, scale=1, input_frames=3, space_frames=5, prediction_frame=5):

    plt.figure(figsize = (10, 30))

    with h5py.File(path_hdf5, "r") as f:
              
        image_list = []

        for i in range(input_frames):
            
            image = int_to_float(f['train/{:0>5d}/depth/depth_{:0>3d}'.format(scene, frame + (i * space_frames))][:], 16)
            image = rescale(image=image, scale=scale, multichannel=True, order=1, anti_aliasing=False)
            image_list.append(image)

            ax = plt.subplot(1, input_frames + 2, i+1)
            plt.imshow(np.reshape(image, (image.shape[0], image.shape[1])), cmap="viridis", vmin=0, vmax=1)
        
        future = int_to_float(f['train/{:0>5d}/depth/depth_{:0>3d}'.format(scene, frame + (input_frames - 1) * space_frames + prediction_frame)][:], 16)
        future = rescale(image=future, scale=scale, multichannel=True, order=1, anti_aliasing=False)

    ax = plt.subplot(1, input_frames + 2, input_frames + 1)
    image_input = np.concatenate(image_list, axis=-1)
    image_input = np.reshape(image_input, (1, image_input.shape[2], image_input.shape[0], image_input.shape[0], 1))
    prediction = model(image_input)
    plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2])), cmap="viridis", vmin=0, vmax=1)

    ax = plt.subplot(1, input_frames + 2, input_frames + 2)
    plt.imshow(np.reshape(future, (future.shape[0], future.shape[1])), cmap="viridis", vmin=0, vmax=1)
    
    plt.show()

def unet_depth(path_hdf5, model, scene=1, frame=50, scale=1, input_frames=4, space_frames=5, prediction_frame=5):

    plt.figure(figsize = (10, 60))

    with h5py.File(path_hdf5, "r") as f:
              
        image_list = []

        for i in range(input_frames):
            
            image = int_to_float(f['train/{:0>5d}/depth/depth_{:0>3d}'.format(scene, frame + (i * space_frames))][:], 16)
            image = rescale(image=image, scale=scale, multichannel=True, order=1, anti_aliasing=False)
            image_list.append(image)

            ax = plt.subplot(1, input_frames + 2, i+1)
            plt.imshow(np.reshape(image, (image.shape[0], image.shape[1])), cmap="viridis", vmin=0, vmax=1)
        
        future = int_to_float(f['train/{:0>5d}/depth/depth_{:0>3d}'.format(scene, frame + (input_frames - 1) * space_frames + prediction_frame)][:], 16)
        future = rescale(image=future, scale=scale, multichannel=True, order=1, anti_aliasing=False)

    ax = plt.subplot(1, input_frames + 2, input_frames + 1)
    image_input = np.concatenate(image_list, axis=-1)
    image_input = np.reshape(image_input, (1, image_input.shape[0], image_input.shape[1], image_input.shape[2]))
    prediction = model(image_input)
    plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2])), cmap="viridis", vmin=0, vmax=1)

    ax = plt.subplot(1, input_frames + 2, input_frames + 2)
    plt.imshow(np.reshape(future, (future.shape[0], future.shape[1])), cmap="viridis", vmin=0, vmax=1)
    
    plt.show()

def unet_seg(path_hdf5, model, scene=1, frame=50, scale=1, input_frames=4, space_frames=2, prediction_frame=5):

    def mask_oneshot_object(mask_array):
    
        oneshot_array = np.empty((mask_array.shape))
        oneshot_array[mask_array == 3] = 1.0
        oneshot_array[mask_array != 3] = 0.0
    
        return oneshot_array
    
    def mask_oneshot_occlu(mask_array):
    
        oneshot_array = np.empty((mask_array.shape))
        oneshot_array[mask_array == 4] = 1.0
        oneshot_array[mask_array != 4] = 0.0
    
        return oneshot_array
    
    def mask_oneshot(mask_array):
    
        oneshot_array = np.zeros((mask_array.shape))
        oneshot_array[mask_array == 3] = 1.0
        oneshot_array[mask_array == 4] = 0.5
    
        return oneshot_array

    plt.figure(figsize = (20, 60))

    with h5py.File(path_hdf5, "r") as f:
              
        image_list = []

        for i in range(input_frames):
            
            image = f['train/{:0>5d}/masks/masks_{:0>3d}'.format(scene, frame + (i * space_frames))][:]

            mask = mask_oneshot(image)
            ax = plt.subplot(1, input_frames + 2, i+1)
            plt.imshow(np.reshape(mask, (mask.shape[0], mask.shape[1])), cmap="viridis", vmin=0, vmax=1)
            
            image = rescale(image=mask, scale=scale, multichannel=True, order=1, anti_aliasing=False)
            image_list.append(image)
        
        future = mask_oneshot_object(f['train/{:0>5d}/masks/masks_{:0>3d}'.format(scene, frame + (input_frames - 1) * space_frames + prediction_frame)][:])
        future = rescale(image=future, scale=scale, multichannel=True, order=1, anti_aliasing=False)

    ax = plt.subplot(1, input_frames + 2, input_frames + 1)
    image_input = np.concatenate(image_list, axis=-1)
    image_input = np.reshape(image_input, (1, image_input.shape[0], image_input.shape[1], image_input.shape[2]))
    prediction = model(image_input)
    plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2])), cmap="viridis", vmin=0, vmax=1)

    ax = plt.subplot(1, input_frames + 2, input_frames + 2)
    plt.imshow(np.reshape(future, (future.shape[0], future.shape[1])), cmap="viridis", vmin=0, vmax=1)
    
    plt.show()

def prediction_image_short(path_hdf5, model, scene=1, frame=50, scale=1, memory_frames=3):

    with h5py.File(path_hdf5, "r") as f:
              
        image_list = []

        for i in range(memory_frames):
            
            image = f['train/{:0>5d}/scene/scene_{:0>3d}'.format(scene, frame - i*5)][:]
            image_list.append(image)
        
        future = f['train/{:0>5d}/scene/scene_{:0>3d}'.format(scene, frame + 5)][:]

    plt.figure(figsize = (10, 30))
    
    ax = plt.subplot(1, 3, 1)
    image = int_to_float(image, 8)
    image = rescale(image=image_list[0], scale=scale, multichannel=True, order=1, anti_aliasing=True)
    plt.imshow(image)

    ax = plt.subplot(1, 3, 2)
    future = int_to_float(future, 8)
    future = rescale(image=future, scale=scale, multichannel=True, order=1, anti_aliasing=True)
    plt.imshow(future)

    ax = plt.subplot(1, 3, 3)
    image_input = np.concatenate(image_list, axis=-1)
    image_input = int_to_float(image_input, 8)
    image_input = rescale(image=image_input, scale=scale, multichannel=True, order=1, anti_aliasing=True)
    image_input = np.reshape(image_input, (1, image_input.shape[0], image_input.shape[1], image_input.shape[2]))
    prediction = model.predict(x=image_input)
    plt.imshow(np.reshape(prediction, (prediction.shape[1], prediction.shape[2], 3)))
    
    plt.show()

def plot_loss(model_train, epochs, identification, loss_name, save):

    train_epochs = range(epochs)
    train_loss = model_train.history["loss"]
    train_val_loss = model_train.history["val_loss"]
            
    plt.figure(figsize = (5, 5))

    ax = plt.subplot(1, 1, 1)
    plt.plot(train_epochs, train_loss, label = loss_name, color = "royalblue")
    plt.plot(train_epochs, train_val_loss, label = "Validation " + loss_name, color = "darkturquoise")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(loss_name)

    plt.show

    if save:
        plt.savefig("./plots/" + identification + ".png")
    

def plot_metrics(model_train, epochs, identification,):

    plot_epochs = range(epochs)
    plot_metrics = []
    plot_val_metrics = []
    for metric in metrics:
        plot_metrics.append(model_train.history[metric])
        plot_val_metrics.append(model_train.history["val_" + metric])

    ax = plt.subplot(1, 3, 2)
    plt.plot(plot_epochs, plot_metric1, label = metric1.upper(), color = "seagreen")
    plt.plot(plot_epochs, plot_val_metric1, label = "Validation " + metric1.upper(), color = "limegreen")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(metric1.upper())

    ax = plt.subplot(1, 3, 3)
    plt.plot(plot_epochs, plot_metric2, label = metric2.upper(), color = "firebrick")
    plt.plot(plot_epochs, plot_val_metric2, label = "Validation " + metric2.upper(), color = "tomato")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(metric2.upper())