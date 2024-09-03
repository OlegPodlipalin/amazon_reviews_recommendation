import os
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from config import MODEL_PATH, MODEL_NAME


class Model:
    def __init__(self):
        self.logger = logging.getLogger("AppLogger")
        self.model = None
        
        self._load_model()
        self.logger.info(f"instance of {self.__class__.__name__} class successfully initialized")
    
    def _load_model(self):
        """Loads pretrained tensorflow model

        Raises:
            e: if loading attempt fails
        """
        model_path = os.path.join(MODEL_PATH, MODEL_NAME)
        self.logger.debug(f"loading model from {model_path}")
        try:
            self.model = load_model(model_path)
            self.logger.info(f"model successfully loaded from {model_path}")
            self.logger.debug(f"model summary: {self.model.summary()}")
        except Exception as e:
            self.logger.error(f"model was not loaded due to exception: {e}")
            raise e
    
    def predict(self, data: Tuple[np.ndarray | pd.DataFrame]) -> np.ndarray:
        """Runs predictions on the given data

        Args:
            data (Tuple[np.ndarray  |  pd.DataFrame]): a tuple that contains user_review_embedding, user_item_embedding, 
            item_description_embedding, and item_price

        Returns:
            np.ndarray: an array of predicted probabilities
        """
        self.logger.debug("running predictions")
        predictions = self.model.predict(data)
        self.logger.info("prediction process successful")
        return predictions
    
    def reload_model(self):
        """Forcedly reloads the pretrained tensorflow model
        """
        self.logger.info("reloading model")
        self._load_model()
