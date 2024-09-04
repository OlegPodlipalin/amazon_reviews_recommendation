#!/bin/bash

download_from_google_drive() {
    local file_id=$1
    local file_dir=$2
    local file_name=$3
    local save_path="${file_dir}/${file_name}"

    local url="https://drive.google.com/uc?export=download&id=${file_id}"

    echo "Downloading ${file_name} to ${save_path}..."
    gdown "$file_id" --output "$save_path"
}

# Download each file into its respective folder
download_from_google_drive "11XkUdutydCgVJ64t74fBxeFVhIsfCXHH" "embeddings" "user_review_emb_df.pq"
download_from_google_drive "1aWScZk_EEVmLbt0V5OvUGyK8zGKTNcoF" "embeddings" "user_item_emb_df.pq"
download_from_google_drive "17QUIqYcsvVlX4EDdaxa5t6PW5pf80Myq" "embeddings" "item_price_df.pq"
download_from_google_drive "16U2HG8C3sMgv3YWqwL18oRaIjAqBe8oX" "embeddings" "item_desc_emb_df.pq"
download_from_google_drive "1ZgCw-iRZ3SQwFXA9h9M22nkiN7QcBs0o" "data" "clean_df.pq"
download_from_google_drive "1AQuABwQtkUJVhxWkEeUMZe_ZcI5VM3mH" "models" "best_model.keras"

echo "All files downloaded successfully."
