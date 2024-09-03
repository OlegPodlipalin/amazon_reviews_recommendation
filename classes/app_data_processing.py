import os
import logging
from typing import Tuple, List
import numpy as np
import pandas as pd
from config import (
    DATA_PATH,
    DF_NAME,
    EMBEDDINGS_PATH,
    IDEDF_NAME,
    IPDF_NAME,
    UREDF_NAME,
    UIEDF_NAME,
    TOP_N,
    COLD_DAYS_WINDOW
)

class AppDataProcessor:
    def __init__(self):
        self.logger = logging.getLogger("AppLogger")
        self._ide_df = None
        self._ip_df = None
        self._ure_df = None
        self._uie_df = None
        self._df = None
        self.cold_start_recommendations = []

        self._load_embeddings()
        self._load_clean_data()
        self.logger.info(f"instance of {self.__class__.__name__} class successfully initialized")

    def _load_embeddings(self):
        """Load the embedding data: 
        - item_desc_emb_df
        - item_price_df
        - user_review_emb_df
        - user_item_emb_df

        Raises:
            e: if loading attempt fails
        """
        self.logger.debug(f"loading embeddings from {EMBEDDINGS_PATH}")
        try:
            self._ide_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, IDEDF_NAME))
            self._ip_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, IPDF_NAME))
            self._ure_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, UREDF_NAME))
            self._uie_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, UIEDF_NAME))
            self.logger.info(f"embeddings successfully loaded from {EMBEDDINGS_PATH}")
            self.logger.debug("embeddings loaded with the shape:") 
            self.logger.debug(f"ide_df: {self._ide_df.shape}, ip_df: {self._ip_df.shape}, ure_df: {self._ure_df.shape}, uie_df: {self._uie_df.shape}")
        except Exception as e:
            self.logger.error(f"embeddings were not loaded due to exception: {e}")
            raise e

    def _load_clean_data(self):
        """Load the clean data (clean_df)

        Raises:
            e: if loading attempt fails
        """
        data_path = os.path.join(DATA_PATH, DF_NAME)
        self.logger.debug(f"loading clean data from {data_path}")
        try:
            self._df = pd.read_parquet(data_path)
            self.logger.info(f"clean data successfully loaded from {data_path}")
            self.logger.debug(f"clean data loaded with the shape: {self._df.shape}")
        except Exception as e:
            self.logger.error(f"clean data was not loaded due to exception: {e}")
            raise e
        self._get_cold_start_recommendations()
    
    def _get_cold_start_recommendations(self):
        """Calculates the list of the most popular high scored products within the last number of days (specified in config.py)
        """
        popular = self._df[self._df["rating"] == 5].groupby("itemName")["reviewTime"].agg(["size", "max"])
        start_date = popular["max"].max() - pd.Timedelta(days=COLD_DAYS_WINDOW)
        self.cold_start_recommendations = popular[popular["max"] > start_date].sort_values("size", ascending=False)[:TOP_N].index.to_list()
        self.logger.info(f"cold start recommendations actual for the last {COLD_DAYS_WINDOW} days extracted")
        self.logger.debug(f"cold start recommendations: {self.cold_start_recommendations}")

    def known_user(self, user: str) -> bool:
        """Checks if the user appears in the list of known users (train data used)

        Args:
            user (str): user name to check

        Returns:
            bool: True if user appears, False otherwise
        """
        search_result = user in self._ure_df.index
        self.logger.debug(f"user: {user} found in the list of known users: {search_result}")
        return search_result

    def get_user_data(self, user: str) -> Tuple[np.ndarray|pd.DataFrame]:
        """Creates a tuple of model input ready data for one user

        Args:
            user (str): user name to prepare data for

        Returns:
            Tuple[np.ndarray|pd.DataFrame]: a tuple that contains user_review_embedding, user_item_embedding, 
            item_description_embedding, and item_price
        """
        self.logger.debug(f"preparing data for user: {user}")
        user_review_emb = np.tile(self._ure_df.loc[user], (len(self._ide_df), 1))
        user_item_emb = np.tile(self._uie_df.loc[user], (len(self._ide_df), 1))
        data = (user_review_emb, user_item_emb, self._ide_df, self._ip_df)
        self.logger.debug(f"data for user {user} ready")
        return data

    def extract_recommendations(self, predictions: np.ndarray) -> List[str]:
        """Extracts recommendations with the highest predicted probability

        Args:
            predictions (np.ndarray): model predictions

        Returns:
            List[str]: list or items with the highest predicted probability
        """
        self.logger.debug(f"extracting top {TOP_N} recommendations")
        top_indices = np.argsort(-predictions[:, 0])[:TOP_N]
        recommendations = self._ide_df.index[top_indices].tolist()
        self.logger.info(f"top recommendations: {recommendations}")
        return recommendations

    def get_random_users(self, n: int) -> List[str]:
        """Randomly samples user names for the list of known users (train data used)

        Args:
            n (int): number of names to sample

        Returns:
            List[str]: list with randomly sampled user names
        """
        self.logger.debug(f"extracting {n} random users")
        sampled_names = np.random.choice(self._ure_df.index, size=n, replace=False)
        users = sampled_names.tolist()
        self.logger.info(f"extracted {n} random users: {users}")
        return users

    def reload_embeddings(self):
        """Forcedly reloads the embeddings data
        """
        self.logger.info("reloading embedding data")
        self._load_embeddings()

    def reload_clean_data(self):
        """Forcedly reloads the clean data
        """
        self.logger.info("reloading clean data")
        self._load_clean_data()
