from tensorflow.keras.applications.resnet50 import ResNet50

def base_model():
    return ResNet50(weights=None, input_shape=(280,280,3), classes=1)

