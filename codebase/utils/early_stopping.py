# early_stopping.py
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor="val_seg", mode="min", verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            monitor (str): What to monitor, e.g., "val_seg", "val_recon", or "val_total".
            mode (str): "min" for loss-based, "max" for accuracy-based.
            verbose (bool): Print info when early stopping is triggered.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def check(self, current_value):
        improvement = (
            (current_value < self.best_score - self.min_delta) if self.mode == "min"
            else (current_value > self.best_score + self.min_delta)
        )

        if improvement:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ Early stopping counter: {self.counter}/{self.patience} epochs without improvement.")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"⏹️ Early stopping triggered: No improvement in '{self.monitor}' for {self.patience} epochs.")
