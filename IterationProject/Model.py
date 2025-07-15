from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, MaxPooling1D 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers



# Classification
def build_LSTM_classifier(
    input_shape, 
    num_classes, 
    ohe=True,
    lstm_units=[32, 16],
    dense_units=[32],
    dropout_rate=0.2,
    use_batchnorm=False,
    l2_reg=1e-4,
    learning_rate=1e-4
    ):

    model = Sequential()

    # LSTM Layers
    for idx, units in enumerate(lstm_units):
        return_seq = (idx < len(lstm_units)-1)
        if idx == 0:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=input_shape,
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

    # Model Compile
    if ohe: loss_function = 'categorical_crossentropy' 
    else: loss_function = 'sparse_categorical_crossentropy'
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )

    return model
    
  

