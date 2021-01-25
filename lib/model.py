from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
def base_model():
    base_model = ResNet50(weights=None, input_shape=(224,224,3),include_top=False, classes=1)
    headModel = base_model.output
    headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)
    return Model(inputs=base_model.input, outputs=headModel)

