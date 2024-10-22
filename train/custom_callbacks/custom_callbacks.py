import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow.keras.backend as K


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Access the learning rate using Keras backend
        lr = K.get_value(self.model.optimizer.learning_rate)
        tf.summary.scalar('learning_rate', data=lr, step=epoch)


class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, test_images):
        super(ImageLogger, self).__init__()
        self.log_dir = log_dir
        self.test_images = test_images[:16]
        if len(self.test_images.shape) == 3:
            self.test_images = tf.expand_dims(self.test_images, -1)  # Adds a channel dimension if not present

    def on_epoch_end(self, epoch, logs=None):
        # Log images to TensorBoard
        file_writer = tf.summary.create_file_writer(self.log_dir)
        if len(self.test_images.shape) == 3:
            self.test_images = tf.expand_dims(self.test_images, -1)  # Adds a channel dimension if not present

        with file_writer.as_default():
            tf.summary.image("Test Images", self.test_images, step=epoch)

class ConfusionMatrixLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_data, log_dir):
        super(ConfusionMatrixLogger, self).__init__()
        self.val_data = val_data
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_labels = self.val_data
        predictions = self.model.predict(val_images)
        cm = confusion_matrix(np.argmax(val_labels, axis=1), np.argmax(predictions, axis=1))
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.text("Confusion Matrix", str(cm), step=epoch)

class F1ScoreLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super(F1ScoreLogger, self).__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_labels = self.val_data
        predictions = self.model.predict(val_images)
        f1 = f1_score(np.argmax(val_labels, axis=1), np.argmax(predictions, axis=1), average='macro')
        print(f"F1 Score for epoch {epoch + 1}: {f1:.4f}")



