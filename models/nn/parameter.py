import numpy as np
import uuid

class Parameter:
    def __init__(self, value: np.ndarray, name: str = "parameter"):
        self._id = uuid.uuid4()
        self.name = name
        self.value = value
        self.gradient = np.zeros_like(value)

    @property
    def id(self):
        return self._id
