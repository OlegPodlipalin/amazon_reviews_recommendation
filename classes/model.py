import os
import logging
from tensorflow.keras.models import load_model
from config import MODEL_PATH, MODEL_NAME


class Model:
    def __init__(self):
        self.logger = logging.getLogger("AppLogger")
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        self.model = load_model(os.path.join(MODEL_PATH, MODEL_NAME))
    
    def predict(self, data):
        self.logger.info("do predictions")
        return self.model.predict(data)