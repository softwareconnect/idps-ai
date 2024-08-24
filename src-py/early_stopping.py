from tensorflow.keras.callbacks import Callback

class MultiMetricEarlyStopping(Callback):
    def __init__(self, val_loss_patience=10, val_accuracy_patience=10, min_delta=0.001, restore_best_weights=True):
        super(MultiMetricEarlyStopping, self).__init__()
        self.val_loss_patience = val_loss_patience
        self.val_accuracy_patience = val_accuracy_patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait_loss = 0
        self.wait_accuracy = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.stopped_epoch = 0
        self.best_weights = None
        self.early_stopped = False  # Flag to check if early stopping occurred

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.wait_loss = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait_loss += 1

        if val_accuracy > self.best_val_accuracy + self.min_delta:
            self.best_val_accuracy = val_accuracy
            self.wait_accuracy = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait_accuracy += 1

        if self.wait_loss >= self.val_loss_patience and self.wait_accuracy >= self.val_accuracy_patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.early_stopped = True  # Set the flag to True
            if self.restore_best_weights:
                print(f"Restoring model weights from the best epoch: {self.stopped_epoch + 1}")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping triggered")
            if self.early_stopped:
                print("Early stopping was triggered.")