import os
import logging
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

    def _load_embeddings(self):
        self._ide_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, IDEDF_NAME))
        self._ip_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, IPDF_NAME))
        self._ure_df =pd.read_parquet(os.path.join(EMBEDDINGS_PATH, UREDF_NAME))
        self._uie_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, UIEDF_NAME))

    def _load_clean_data(self):
        self._df = pd.read_parquet(os.path.join(DATA_PATH, DF_NAME)) 
        self._get_cold_start_recommendations()
    
    def _get_cold_start_recommendations(self):
        popular = self._df[self._df["rating"] == 5].groupby("itemName")["reviewTime"].agg(["size", "max"])
        start_date = popular["max"].max() - pd.Timedelta(days=COLD_DAYS_WINDOW)
        self.cold_start_recommendations = popular[popular["max"] > start_date].sort_values("size", ascending=False)[:TOP_N].index.to_list()

    def known_user(self, user):
        return user in self._ure_df.index

    def get_user_data(self, user):
        self.logger.debug("preparing user data")
        user_review_emb = np.tile(self._ure_df.loc[user], (len(self._ide_df), 1))
        user_item_emb = np.tile(self._uie_df.loc[user], (len(self._ide_df), 1))
        return (user_review_emb, user_item_emb, self._ide_df, self._ip_df)

    def extract_recommendations(self, predictions):
        top_indices = np.argsort(-predictions[:, 0])[:TOP_N]
        return self._ide_df.index[top_indices].tolist()

    def get_random_users(self, n):
        sampled_names = np.random.choice(self._ure_df.index, size=n, replace=False)
        return sampled_names.tolist()

    def reload_embeddings(self):
        self._load_embeddings()

    def reload_clean_data(self):
        self._load_clean_data()
        
        
# def load_embeddings():
#     ide_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, IDEDF_NAME))
#     ip_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, IPDF_NAME))
#     ure_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, UREDF_NAME))
#     uie_df = pd.read_parquet(os.path.join(EMBEDDINGS_PATH, UIEDF_NAME))
#     return ide_df, ip_df, ure_df, uie_df


# def load_clean_data():
#     df = pd.read_parquet(os.path.join(DATA_PATH, DF_NAME))
#     return df


# ide_df, ip_df, ure_df, uie_df = load_embeddings()
# df = load_clean_data()


# def known_user(user):
#     return user in ure_df.index


# def get_user_data(user):
#     user_review_emb = np.tile(ure_df.loc[user], (len(ide_df), 1))
#     user_item_emb = np.tile(uie_df.loc[user], (len(ide_df), 1))
#     return (user_review_emb, user_item_emb, ide_df, ip_df)


# def extract_recommendations(predictions):
#     top_indices = np.argsort(-predictions[:, 0])[:TOP_N]  
#     return ide_df.index[top_indices].tolist()


# def get_cold_start_recommendations():
#     popular = df[df["rating"] == 5].groupby("itemName")["reviewTime"].agg(["size", "max"])
#     start_date = popular["max"].max() - pd.Timedelta(days=COLD_DAYS_WINDOW)
#     return popular[popular["max"] > start_date].sort_values("size", ascending=False)[:TOP_N].index.to_list()


# def get_random_users(n):
#     sampled_names = np.random.choice(ure_df.index, size=n, replace=False)
#     return sampled_names.tolist()


# def reload_embeddings():
#     global ide_df, ip_df, ure_df, uie_df
#     ide_df, ip_df, ure_df, uie_df = load_embeddings()


# def reload_clean_data():
#     global df
#     df = load_clean_data()
