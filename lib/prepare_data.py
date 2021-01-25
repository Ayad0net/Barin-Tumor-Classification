from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH_IMAGES = "dataset/brain-data"



image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

train_dataset = image_generator.flow_from_directory(batch_size=16,
                                                seed=101,
                                                 directory=PATH_IMAGES,
                                                 shuffle=True,
                                                 target_size=(224, 224), 
                                                 subset="training",
                                                 class_mode='binary')

validation_dataset = image_generator.flow_from_directory(batch_size=16,
                                                seed=101,
                                                 directory=PATH_IMAGES,
                                                 shuffle=True,
                                                 target_size=(224, 224), 
                                                 subset="validation",
                                                 class_mode='binary')


def load_data():
    print("loading start, please wait ...")
    
    return train_dataset, validation_dataset
