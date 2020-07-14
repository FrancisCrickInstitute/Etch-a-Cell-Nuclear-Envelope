import csv
import six
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.keras.callbacks import Callback


# Modification of the following is necessary to enable training history logging.
# Maybe this will be fixed with TF 2.0, in which case, delete this.


@tf_export('keras.callbacks.CSVLogger')
class CSVLoggerMODIFIED(Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    Example:

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    Arguments:
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        super(CSVLoggerMODIFIED, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            self.mode = 'a'
        else:
            self.mode = 'w'

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            writer = csv.writer(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_ALL)
            writer.writerow([epoch, logs['val_loss'], logs['loss'], logs['lr']])

    def on_train_end(self, logs=None):
        self.writer = None