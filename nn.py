from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf


class CNN(Sequential):
    def __init__(self, conv_filters_list, fc_nodes_list, n_features, time_history, output_nodes, init_lr=1e-5,
                 loss="mse", dropout_rate=0.2):
        super(CNN, self).__init__()  # Call parent class' constructor

        # Convolutional segment
        for conv_layer in range(len(conv_filters_list)):
            self.add(Conv2D(conv_filters_list[conv_layer], (1, 2), padding="same", activation="relu",
                            input_shape=(time_history, n_features, 1)))
            self.add(MaxPooling2D())
        self.add(Flatten())
        # Fully-connected segment
        for fc_layer in range(len(fc_nodes_list)):
            self.add(Dense(fc_nodes_list[fc_layer], activation="relu"))
            self.add(Dropout(dropout_rate))

        # Output layer
        self.add(Dense(output_nodes, activation="linear"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
        self.compile(optimizer=optimizer, loss=loss)

    def early_stop(self, early_stop_patience=25):
        return EarlyStopping(monitor="val_loss", verbose=1, patience=early_stop_patience)