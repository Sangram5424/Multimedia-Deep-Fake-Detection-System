import numpy as np
import os
import gc  # Garbage collector for cleaning deleted data from memory
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tqdm import tqdm  # Used for creating progress bar

def load_images_from_directory(directory, img_cols, img_rows):
    img_list = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('L').resize((img_cols, img_rows))
                img_array = np.array(img).flatten()
                img_list.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(img_list, dtype='float32')

def preprocess_data(fall_dir, notfall_dir, img_cols, img_rows):
    fall_images = load_images_from_directory(fall_dir, img_cols, img_rows)
    notfall_images = load_images_from_directory(notfall_dir, img_cols, img_rows)
    
    Mainmatrix = np.vstack((fall_images, notfall_images))
    num_samples = Mainmatrix.shape[0]
    
    labels = np.zeros((num_samples,), dtype=int)
    labels[:len(fall_images)] = 1  # Label falls as 1
    labels[len(fall_images):] = 0  # Label non-falls as 0
    
    data, Label = shuffle(Mainmatrix, labels, random_state=2)
    return data, Label

def create_cnn_model(img_cols, img_rows):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_cols, img_rows, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    img_cols, img_rows = 64, 64
    fall_dir = "E:/2025 Data/ sangram code/BG3 Groups/Deepfake/deepfake/abnormal1"
    notfall_dir = "E:/2025 Data/ sangram code/BG3 Groups/Deepfake/deepfake/normal1"
    
    data, labels = preprocess_data(fall_dir, notfall_dir, img_cols, img_rows)
    
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=1)
    
    X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
    X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)
    
    X_train /= 255
    X_test /= 255
    
    Y_train = np_utils.to_categorical(Y_train, 2)
    Y_test = np_utils.to_categorical(Y_test, 2)
    
    model = create_cnn_model(img_cols, img_rows)
    
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)
    
    MODEL_NAME = "model11.h5"
    model.save(MODEL_NAME)
    
    predictions = model.predict(X_train)
    
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(Y_train, axis=1)) * 100
    result = f"Training Accuracy is {accuracy:.2f}%"
    
    print(result)
    return result

if __name__ == "__main__":
    main()





















# def main():
    
#     #Import some packages to use
#     import numpy as np
    
#     #To see our directory
#     import os
#     import gc   #Gabage collector for cleaning deleted data from memory
#     from PIL import Image
#     from tqdm import tqdm # Used for creating progress bar
    
#     img_cols, img_rows = 64,64
    
    
    
#     vid_dir=[]
#     Fall_dir="E:/2023 Data/Softech mayuri code/Mayuri Groups/Mayuri Groups/23SS312-Deepfake/deepfake/abnormal1"
    
#     vid_dir = []
#     for root, dirs, files in os.walk(Fall_dir):
#         for i in range(len(files)):
#             vid_dir.append(root + '/' + files[i])
    
    
    
    
#     immatrix = np.array([np.array(Image.open(im2).convert('L').resize((img_cols, img_rows))).flatten()
#                   for im2 in vid_dir],'f')
    
#     ######FOR NOT event22222222222222222222222222222222222222222222222222222222222222222222222222222222
#     del vid_dir
#     gc.collect()
    
#     vid_dir=[]
#     NotFall_dir=r'E:/2023 Data/Softech mayuri code/Mayuri Groups/Mayuri Groups/23SS312-Deepfake/deepfake/normal1'
    
#     vid_dir = []
#     for root, dirs, files in os.walk(NotFall_dir):
#         for i in range(len(files)):
#             vid_dir.append(root + '/' + files[i])
    
    
    
    
#     immatrix2 = np.array([np.array(Image.open(im2).convert('L').resize((img_cols, img_rows))).flatten()
#                   for im2 in vid_dir],'f')
    
#     del vid_dir
#     gc.collect()
    
#     ####################################################################################################################
    
#     Mainmatrix=np.vstack((immatrix,immatrix2))
    
#     num_samples=Mainmatrix.shape[0]
    
#     label=np.ones((num_samples,),dtype = int)
    
#     label[0:370]=1
#     label[371:]=0
    
#     from sklearn.model_selection import train_test_split
#     from sklearn.utils import shuffle
#     from keras.utils import np_utils
    
    
    
#     data,Label = shuffle(Mainmatrix,label, random_state=2)
#     train_data = [data,Label]
        
#     nb_classes = 2
    
#     (X, y) = (train_data[0],train_data[1])
    
    
#     # STEP 1: split X and y into training and testing sets
    
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
#     X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)
    
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
    
#     X_train /= 255
#     X_test /= 255
    
#     print('y_train shape:', Y_train.shape)
#     print('X_train shape:', X_train.shape)
#     print(X_train.shape[0], 'train samples')
#     print(X_test.shape[0], 'test samples')
        
#     # convert class vectors to binary class matrices
#     Y_train = np_utils.to_categorical(Y_train, nb_classes)
#     Y_test = np_utils.to_categorical(Y_test, nb_classes)
        
    
#     #######################################################################################################
#     from keras.models import Sequential
#     from keras.models import Sequential
#     from keras.layers import Convolution2D
#     from keras.layers import MaxPooling2D
#     from keras.layers import Flatten
#     from keras.layers import Dense, Dropout
#     from keras import optimizers
    
#     from keras.layers import Dense, Conv2D, Flatten
    
#     MODEL_NAME="model11.h5"
    
    
#     #####################CNN basic MODEL#############################################
#     model = Sequential()
    
#     model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_cols, img_rows,1)))
    
#     model.add(Conv2D(32, kernel_size=3, activation='relu'))
    
#     model.add(Flatten())
    
#     model.add(Dense(2, activation='softmax'))
    
    
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
#     model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)
    
#     model.save(MODEL_NAME)
    
#     predictions = model.predict(X_train)
    
#     ##################################################################################
    
    
#     #  # Initialing the CNN
#     # classifier = Sequential()
    
#     # # Step 1 - Convolutio Layer 
#     # classifier.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 1), activation = 'relu'))
    
#     # #step 2 - Pooling
#     # classifier.add(MaxPooling2D(pool_size =(2,2)))
    
#     # # Adding second convolution layer
#     # classifier.add(Convolution2D(32, 3,  3, activation = 'relu'))
#     # classifier.add(MaxPooling2D(pool_size =(2,2)))
    
#     # #Adding 3rd Concolution Layer
#     # classifier.add(Convolution2D(64, 3,  3, activation = 'relu'))
#     # classifier.add(MaxPooling2D(pool_size =(2,2)))
    
    
#     # #Step 3 - Flattening
#     # classifier.add(Flatten())
    
#     # #Step 4 - Full Connection
#     # classifier.add(Dense(256, activation = 'relu'))
#     # classifier.add(Dropout(0.2))
#     # classifier.add(Dense(2, activation = 'softmax'))  #change class no.
    
#     # #Compiling The CNN
#     # classifier.compile(
#     #               optimizer = optimizers.SGD(lr = 0.01),
#     #               loss = 'categorical_crossentropy',
#     #               metrics = ['accuracy'])
        
    
    
#     # classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)
    
    
#     # classifier.save(MODEL_NAME)
#     # predictions = classifier.predict(X_train)
    
#     ############################################################################################################
    
#     accuracy = 0
#     for prediction, actual in zip(predictions, Y_train):
#         predicted_class = np.argmax(prediction)
#         actual_class = np.argmax(actual)
#         if(predicted_class == actual_class):
#             accuracy+=1
    
#     accuracy =( accuracy / len(Y_train))*100
    
#     A = "Training Accuracy is {0}".format(accuracy)
        
#     return A
    
#     ##########################################################################################################
    
    
    
#     # from keras.models import load_model
    
#     # img_cols, img_rows = 64,64
    
#     # FALLModel=load_model(r'D:\Alka_python_2019_20\FDD\Fall-Detection-with-CNNs-and-Optical-Flow-master\25APRFALLDtection.h5')
                         
#     # # vid_dir="D:/Alka_python_2019_20/FDD/Fall-Detection-with-CNNs-and-Optical-Flow-master/URFD_images/Falls/fall_fall-01"
#     # # vid_dir=r'D:\Alka_python_2019_20\FDD\Fall-Detection-with-CNNs-and-Optical-Flow-master\URFD_images\NotFalls\notfall_fall-01_pre'
    
#     # vid_dir=r"C:\alka\FULL_FDD_Data\RGB\fall-01-cam0-rgb"
#     # # vid_dir=r"D:\Alka_python_2019_20\FDD\Fall-Detection-with-CNNs-and-Optical-Flow-master\URFD_images\NotFalls_full\notfall_adl-40"
     
#     # imlist = os.listdir(vid_dir)
    
#     # immatrix = np.array([np.array(Image.open(vid_dir + '\\' + im2).convert('L').resize((img_cols, img_rows))).flatten()
#     #               for im2 in imlist],'f')
    
    
#     # X_img = immatrix.reshape(immatrix.shape[0], img_cols, img_rows, 1)
    
#     # X_img = X_img.astype('float32')
    
#     # X_img /= 255
    
#     # predicted =FALLModel.predict(X_img)
    
    
#     # for i in range(len(predicted)):
#     #     if predicted[i][0] < 0.5:
#     #         predicted[i][0] = 0
#     #         predicted[i][1] = 1
    
#     #     else:
            
#     #         predicted[i][0] = 1
#     #         predicted[i][1] = 0
        
#     #     # Array of predictions 0/1
    
#     # predicted = np.asarray(predicted).astype(int) 
    
