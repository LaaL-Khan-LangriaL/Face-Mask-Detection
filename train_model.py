import keras
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator




train_data_path = '/home/agrivision/Mask Detection/dataset/train'
validation_data_path = '/home/agrivision/Mask Detection/dataset/valid'


#////////////////////////////////////////////////////////////////////////////
#show augmented images 

def plotImages (images_arr):
    fig , axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()



#//////////////////////////////////////////////////////////////////////////    

# Generating more images for training model on dataset
training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range = 40,
                                      width_shift_range = 0.2,
                                      height_shift_range = 0.2,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True,
                                      fill_mode = 'nearest')

# this will read pictures found in train folder 
#and will generate batches of augmented images data

training_data = training_datagen.flow_from_directory(train_data_path,
                                                     target_size = (200, 200),
                                                     batch_size = 128,
                                                     class_mode = 'binary')

#////////////////////////////////////////////////////////////////////////////




#////////////////////////////////////////////////////////////////////////

#rescaling validation data set but not augmenting this data

valid_datagen = ImageDataGenerator(rescale = 1./255)

validation_data = valid_datagen.flow_from_directory(validation_data_path,
                                               target_size = (200, 200),
                                               batch_size = 128,
                                               class_mode = 'binary')


#//////////////////////////////////////////////////////////////////////////

#plot images 

#images = [training_data[0][0][0] for i in range (5)]
#plotImages(images)

#//////////////////////////////////////////////////////////////////////////

#Model check point for saving best accuracy 

model_path = '/home/agrivision/Mask Detection/face_detect_model.h5'
checkpoint = ModelCheckpoint(model_path, monitor = 'val_accuracy', verbose = 1, save_best_only=True, mode = 'max')
callbacks_list = [checkpoint]

#/////////////////////////////////////////////////////////////////////////////////

#Build a CNN MOdel

cnn_model = keras.models.Sequential([
                                        keras.layers.Conv2D(filters = 32, kernel_size = 5, input_shape = [200, 200, 3]),
                                        keras.layers.MaxPooling2D(pool_size = (4, 4)),
                                        keras.layers.Conv2D(filters = 64, kernel_size = 4),
                                        keras.layers.MaxPooling2D(pool_size = (3, 3)),
                                        keras.layers.Conv2D(filters = 128, kernel_size = 3),
                                        keras.layers.MaxPooling2D(pool_size = (2, 2)), 
                                        keras.layers.Conv2D(filters = 256, kernel_size = 2),
                                        keras.layers.MaxPooling2D(pool_size = (2, 2)),
                                        
                                        keras.layers.Dropout(0.5),
                                        keras.layers.Flatten(),   #neural network building
                                        keras.layers.Dense(units = 128, activation='relu'), #input layers
                                        keras.layers.Dropout(0.1),
                                        keras.layers.Dense(units = 256, activation='relu'),
                                        keras.layers.Dropout(0.25),
                                        keras.layers.Dense(units = 2, activation='softmax')# output layers , unit 2 is due to claases number are two
                                    ])

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Compile and train our model 

cnn_model.compile(optimizer = Adam(lr = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']),

history  = cnn_model.fit(training_data,
                     epochs = 50,
                     verbose = 1,
                     validation_data = validation_data,
                     callbacks = callbacks_list)  # time start 5:35 PM sun-14-March Pakistan Islamabad 
