from tensorflow.keras.callbacks import ModelCheckpoint

from src.ml.DataLoader import get_train_generator, get_validation_data
from src.ml.CSVLoggerMODIFIED import CSVLoggerMODIFIED
from src.ml.ReduceLROnPlateauMODIFIED import ReduceLROnPlateauMODIFIED


def get_callbacks(model_name):
    callbacks = list()
    model_filename = model_name + '__{epoch}.hdf5'
    callbacks.append(ModelCheckpoint(
        filepath=model_filename,
        verbose=1,
        monitor='loss',
        save_best_only=False,
        period=1,
        mode='min'
    ))

    callbacks.append(ReduceLROnPlateauMODIFIED(
        monitor='val_loss',
        factor=0.75,  # lr = lr*factor
        patience=4,  # how many epochs no change
        verbose=1
    ))

    log_filename = model_name + '.log'
    callbacks.append(CSVLoggerMODIFIED(log_filename, append=True))

    return callbacks


def model_train(model, params, data_loader, epoch=0):
    model_params = params['model']
    model_name = data_loader.get_name()
    batch_size = model_params['batch_size']
    epochs = model_params['epochs']
    steps_per_epoch = model_params['steps_per_epoch']

    model.fit_generator(
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        initial_epoch=epoch,
        generator=get_train_generator(data_loader, batch_size),
        validation_data=get_validation_data(data_loader),
        callbacks=get_callbacks(model_name),
    )

    return model
