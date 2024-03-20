import os
import datetime
import tensorflow as tf
from keras_applications.resnet_1d import ResNet18
from segmentation_models.segmentation_models.models.unet_1d import Unet1D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_confusion_matrix(X_test, Y_test, checkpoint_full_train):

    model = Unet1D(backbone_name = "resnet18_1d", classes = 1, input_shape = (X_test.shape[1], X_test.shape[2]))
    model.compile(
        optimizer = "adam",
        loss = "binary_focal_crossentropy",
        metrics=['accuracy']    
    )

    model = tf.keras.models.load_model(checkpoint_full_train)


    # Calculate confusion matrix
    conf_matrix = confusion_matrix(
            Y_test.reshape(-1).astype(int), 
            model.predict(X_test).reshape(-1).astype(int)
        )

    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()