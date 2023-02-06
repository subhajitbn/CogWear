"""Examples of very simple model architectures that can be trained for
cognitive load and stress detection on blood volume pulse data.
The models take a snippet of 1D signal of length n_timesteps and have
 n_features per time step - e.g. 1 if signal comes from one device.  """

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Reshape, MaxPooling1D, Flatten, Dropout


def baseline_model(n_timesteps, n_features, out, lr):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(out, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy', tf.keras.metrics.AUC(from_logits=False, name='auc')])
    return model


def cogNet(n_timesteps, n_features, out, lr):
    input_img = Input(shape=(n_timesteps, n_features))
    tower_1 = Conv1D(8, 1, padding='same', activation='relu')(input_img)
    tower_1 = Conv1D(8, 3, padding='same', activation='relu')(tower_1)
    tower_2 = Conv1D(8, 1, padding='same', activation='relu')(input_img)
    tower_2 = Conv1D(8, 5, padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling1D(3, strides=1, padding='same')(input_img)
    tower_3 = Conv1D(8, 1, padding='same', activation='relu')(tower_3)
    outputf = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=2)
    conv1 = Conv1D(filters=16, kernel_size=3, activation='relu')(outputf)
    conv2 = Conv1D(filters=8, kernel_size=3, activation='relu')(conv1)
    output = Flatten()(conv2)
    outp = Dense(out, activation='softmax')(output)
    model = Model(inputs=input_img, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy', tf.keras.metrics.AUC(from_logits=False, name='auc')])
    return model


def create_model(model_name, n_timesteps, n_features, out, lr=0.001):
    if model_name == 'baseline':
        return baseline_model(n_timesteps, n_features, out, lr)
    elif model_name == 'cogNet':
        return cogNet(n_timesteps, n_features, out, lr)
    else:
        print("Incorrect model name")
        return None
