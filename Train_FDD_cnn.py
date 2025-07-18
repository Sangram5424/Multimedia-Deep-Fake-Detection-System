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




















