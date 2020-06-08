from ..utils.pytorch import save_model_with_params_from_ckpt

class EarlyStopper:
    def __init__(self, min_epochs, patience, min_delta=1e-4, verbose=False):
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_val_loss = None
        self.best_epoch = None
        self.best_model_name = 'early-stopping-best-model.tar'
        self.do_early_stop = False
        self.epochs = 0
        self.counter = 0

    def register_epoch(self, val_loss, model, model_dir):
        if self.epochs >= self.min_epochs:
            # calculate and store the best model after min_epochs reached
            if self.best_val_loss is None or val_loss <= (self.best_val_loss + self.min_delta):
                self.best_val_loss = val_loss
                self.best_epoch = self.epochs
                save_model_with_params_from_ckpt(model, model_dir, name=self.best_model_name)
                # reset the counter on seeing a smaller val loss
                self.counter = 0
            elif val_loss > self.best_val_loss and self.patience > 0:
                # count number of times the val_loss is more than the best loss so far.
                self.counter += 1

            if self.counter >= self.patience:
                # stop when loss has increased continuously for #patience epochs
                self.do_early_stop = True
        self.epochs += 1
