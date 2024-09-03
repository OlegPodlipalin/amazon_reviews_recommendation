import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

DATA_PATH= os.path.join(PROJECT_PATH, "data")
MODEL_PATH= os.path.join(PROJECT_PATH, "models")
EMBEDDINGS_PATH = os.path.join(PROJECT_PATH, "embeddings")

IDEDF_NAME = "item_desc_emb_df.pq"
IPDF_NAME = "item_price_df.pq"
UREDF_NAME = "user_review_emb_df.pq"
UIEDF_NAME = "user_item_emb_df.pq"
DF_NAME = "clean_df.pq"
MODEL_NAME = "best_model.keras"

TOP_N = 5
COLD_DAYS_WINDOW = 30

os.makedirs(os.path.join(PROJECT_PATH, "logs"), exist_ok=True)
LOG_FILE = os.path.join(PROJECT_PATH, "logs", "logs.log")

