import numpy as np # linear algebra
from keras.models import Sequential
from keras.layers import BatchNormalization , Conv2D , Dense , Flatten , MaxPool2D
#from keras.optimizers import Adam
import tensorflow as tf
#from tensorflow.keras.optimizers import Adam
import valohai
import uuid

def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])

# Read the data
path_dataset = valohai.inputs('dataset').path()

with np.load(path_dataset, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_val, y_val = f['x_val'], f['y_val']

# Train the model
def Mymodel(input_shape , L2):
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu', kernel_regularizer=L2,padding='same',input_shape=input_shape))
    model.add(MaxPool2D((2,2) ,strides=(2,2), padding='same'))
    model.add(Conv2D(64 , kernel_size=(3,3),activation='relu', padding='same'))
    model.add(MaxPool2D((2,2) ,strides=(2,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(64 , activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32 , activation='relu', kernel_regularizer=None))
    model.add(Dense(16 , activation='relu', kernel_regularizer=None))
    model.add(Dense(8 , activation='relu'))
    model.add(Dense(5 , activation='softmax'))
        
    #model.compile(optimizer=Adam(lr=valohai.parameters("learning_rate").value) , loss='categorical_crossentropy' , metrics=['accuracy'])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Print metrics out as JSON
    # This enables Valohai to version your metadata
    # and for you to use it to compare experiments

    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
    model.fit(x_train, y_train, epochs=valohai.parameters('epochs').value, callbacks=[callback])

    # Evaluate the model and print out the test metrics as JSON

    val_loss, val_accuracy = model.evaluate(x_val,  y_val, verbose=2)
    with valohai.logger() as logger:
        logger.log('val_accuracy', val_accuracy)
        logger.log('val_loss', val_loss)


    return model

model = Mymodel((150,150,3), None)
model.summary()

#history=model.fit(x_train,y_train,validation_data=(x_val,y_val),
#                  batch_size=valohai.parameters('batch_size').value,
#                  epochs=valohai.parameters('epochs').value)

# Save the trained model
suffix = uuid.uuid4()
#output_path = valohai.outputs().path(f'model-{suffix}.h5')
dataset_name = valohai.parameters('dataset_name').value
output_path = valohai.outputs().path(f'model-' + dataset_name + '.h5')
model.save(output_path)