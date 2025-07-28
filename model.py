import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

 
 # classifer

def build_lstm_classifier(input_shape, num_classes, oh=True):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    # oh : one hot encoding 사용 여부
    if oh: loss_function = 'categorical_crossentropy'
    else: loss_function = 'sparse_categorical_crossentropy'
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss= loss_function,
        metrics=['accuracy'],
    )
    
    # return model, oh
    return model

def build_lstm_classifier_advanced(
    input_shape,
    num_classes,
    oh=True,
    lstm_units=[64, 32],            # 각 LSTM layer의 유닛 수
    dense_units=[32],              # Dense layer 유닛 수
    dropout_rate=0.3,
    use_batchnorm=False,
    l2_reg=1e-4,
    learning_rate=1e-4
):
    model = Sequential()
    
    # LSTM Layers
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(LSTM(units, return_sequences=return_seq,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Dense Layers
    for units in dense_units:
        model.add(Dense(units, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Loss
    loss_function = 'categorical_crossentropy' if oh else 'sparse_categorical_crossentropy'
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )
    
    # return model, oh
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-Head Attention
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feedforward
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = tf.keras.layers.Add()([x_ff, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def build_transformer_classifier(
    input_shape,
    num_classes,
    oh=False,
    num_transformer_blocks=2,
    head_size=64,
    num_heads=4,
    ff_dim=128,
    dropout=0.3,
    mlp_units=[64],
    l2_reg=0.0,
    learning_rate=1e-4
):
    inputs = Input(shape=input_shape)

    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)
    for units in mlp_units:
        x = Dense(units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Dropout(dropout)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    loss_function = "categorical_crossentropy" if oh else "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    # return model, oh
    return model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes, kernel_size=50, oh=False):
    model = Sequential()
    
    # 1st Conv Block
    model.add(Conv1D(64, kernel_size=kernel_size, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # 2nd ~ 5th Conv Blocks
    for _ in range(4):
        model.add(Conv1D(64, kernel_size=kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
    
    # Flatten and Dense
    model.add(Flatten())
    model.add(Dense(28, activation='relu'))  # 논문: 마지막 conv 출력의 절반
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',  # 정수 라벨이면 sparse
        optimizer=Adam(),
        metrics=['accuracy']
    )

    # return model, oh
    return model

# regressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def build_lstm_regressor_advanced(
    input_shape,
    num_classes,
    lstm_units=[64, 32],            # 각 LSTM layer의 유닛 수
    dense_units=[32],              # Dense layer 유닛 수
    dropout_rate=0.3,
    use_batchnorm=False,
    l2_reg=0.0,
    learning_rate=1e-4
):
    model = Sequential()
    # LSTM Layers
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(LSTM(units, return_sequences=return_seq,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Dense Layers
    for units in dense_units:
        model.add(Dense(units, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output Layer
    # model.add(Dense(num_classes))
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Dense(num_classes, activation='sigmoid'))
    # model.add(Dense(num_classes, activation='tanh'))
    # Loss
    loss_function = 'mse'
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['mae']
    )
    
    return model

