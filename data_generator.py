import tensorflow as tf
import numpy as np
import cv2

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(0)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = image
    return new_image

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, directory, batch_size=32, target_size=(128, 128), shuffle=True):
        'Initialization'
        self.target_size = target_size
        self.batch_size = batch_size
        self.directory = directory

        self.img_paths = [] 
        self.img_paths_wo_ext = []      
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".jpg") or file.lower().endswith(".png") or file.lower().endswith(".jpeg"):
                    noExt = os.path.splitext(os.path.join(root, file))[0]
                    if os.path.isfile(noExt+".txt"):
                      self.img_paths.append(os.path.join(root, file))
                      self.img_paths_wo_ext.append(noExt)

        self.shuffle = shuffle
        self.on_epoch_end()



    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_paths) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_paths = [self.img_paths[k] for k in indexes]
        list_paths_wo_ext = [self.img_paths_wo_ext[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_paths, list_paths_wo_ext)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_paths, list_paths_wo_ext):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size, 8))


        # Generate data
        for i, ID in enumerate(list_paths):
            # Store sample
            file = open(list_paths_wo_ext[i]+".txt", 'r')
            label = file.readlines()[0]
            splitted = label.split(" ")

            img = cv2.imread(ID)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = letterbox_image(img, tuple((self.target_size[0],self.target_size[1])))
            X[i,] = cv2.resize(resized, self.target_size)
            y[i,] = np.array([float(splitted[1]) * self.target_size[0], float(splitted[2]) * self.target_size[1], float(splitted[3]) * self.target_size[0], float(splitted[4]) * self.target_size[1], float(splitted[5]) * self.target_size[0], float(splitted[6]) * self.target_size[1], float(splitted[7]) * self.target_size[0], float(splitted[8]) * self.target_size[1]])


        return X, y