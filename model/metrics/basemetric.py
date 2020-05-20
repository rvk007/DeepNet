class Metric:
    """
    Calculates the mean absolute error.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    """
    def __init__(self, output):
        self._sum_of_errors = 0.0
        self._num_examples = 0
        self.update(output)  

    def reset(self):
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError