from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def track(self, frame):
        pass
