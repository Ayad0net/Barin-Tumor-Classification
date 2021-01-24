from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH_IMAGES = "dataset/brain-data"



image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=PATH_IMAGES,
                                                 shuffle=True,
                                                 target_size=(280, 280), 
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=PATH_IMAGES,
                                                 shuffle=True,
                                                 target_size=(280, 280), 
                                                 subset="validation",
                                                 class_mode='categorical')


def load_data():
    print("loading start, please wait ...")
    
    return train_dataset, validation_dataset
