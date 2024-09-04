# Personalized Recommendation System

## Overview

This project implements a personalized recommendation system that provides top 5 product recommendations for each user, including handling new, unknown customers effectively. The system utilizes a Flask API to manage interactions and uses advanced machine learning techniques to generate recommendations based on user and item interactions.

## Project Structure

```
├── /classes                    # Classes for data processing and model operations
├   ├── app_data_processing.py  # Class for data related application operations
├   └── model.py                # Class for loading and using pretrained model in the application
├── /data                       # Dataset directory
├   ├── amazon_reviews.csv      # <- Here should be placed an unarchived Amazon Reviews 2018 (Full Dataset)
├   └── clean_df.pq             # <- *Here should be placed clean dataset
├── /embeddings                 # Precomputed embeddings for fast loading
├   ├── item_desc_emb_df.pq     # <- *Here should be placed items description embedded dataset
├   ├── item_price_df.pq        # <- *Here should be placed item prices dataset
├   ├── user_item_emb_df.pq     # <- *Here should be placed user item-based embeddings dataset
├   └── user_review_emb_df.pq   # <- *Here should be placed user review-based embeddings dataset
├── /logs                       # Application logs will appear in this directory
├── /models                     # Trained model files
├   └── best_model.keras        # <- *Here should be placed pretrained tensorflow model
├── /report                     # Jupyter notebooks for EDA and model selection
├   ├── eda.ipynb               # Jupyter notebook with exploratory data analysis
├   └── model_selection.ipynb   # Jupyter notebook with data preprocessing and modeling
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── app.py                      # Flask application entry point
├── config.py                   # Configuration parameters for the application
├── download_data.sh            # Script to download and prepare data
├── poetry.lock                 # Poetry package versions lockfile
├── pyproject.toml              # Poetry dependencies and project settings
└── README.md                   # Project documentation

* Use download_data.sh to download all the datasets directly into directories or run report/model_selection.ipynb
```

## Installation

1. Clone this repository.
```bash
git clone https://github.com/OlegPodlipalin/amazon_reviews_recommendation.git
```
2. Ensure that [Poetry](https://python-poetry.org/) is installed on your machine. To install Poetry:
```bash
pip install poetry
```
3. Install dependencies using Poetry:
```bash
poetry install
```
4. Run the download_data.sh script to populate the application with necessary data:
```bash
./download_data.sh
```
or produce this the data during running `model_selection.ipynb` jyputer notebook from `report` folder.
> Note that this option is computationally expensive and can last more than 30 minutes (depending on the machine)

## Running the Application

Execute the following command to start the Flask server:
```bash
poetry run python app.py
```

## API Usage

The API offers several endpoints to interact with the recommendation system:

### Get Random Users
- **Endpoint:** `/random_users`
- **Method:** GET
- **Description:** Retrieves one or more random users from the system.
- **Parameters:**
  - `n` (optional): Specifies the number of random users to retrieve.
- **Example Request**:
```
    GET /random_users?n=5
```
- **Example Response**:
```json
    {
        ["user1", "user2", "user3", "user4", "user5"]
    }
```

### Check User Existence
- **Endpoint:** `/check_users`
- **Method:** GET
- **Description:** Checks if specified users are known in the system.
- **Parameters:**
  - `users`: A comma-separated list of users to check.
- **Example Request**:
```
    GET /check_users?users=user1,user2
```
- **Example Response**:
```json
    {
        "user1": True,
        "user2": False
    }
```

### Get Personalized Recommendations
- **Endpoint:** `/recommendations`
- **Method:** GET
- **Description:** Provides personalized product or content recommendations for specified users.
- **Parameters:**
  - `users`: A comma-separated list of users to get recommendations for.
- **Example Request**:
```
    GET /recommendations?users=user1,user2
```
- **Example Response**:
```json
    {
        "user1": ["product1", "product2"],
        "user2": ["product3", "product4"]
    }
```

### Reload Data
- **Endpoint:** `/reload_data`
- **Method:** POST
- **Description:** Triggers a reload of the underlying data used by the recommendation system. This might be necessary after updating data files or databases.
- **Body:** None required for simple reloads.
- **Example Request**:
```
    POST /reload_data
```
- **Example Response**:
```json
    {
        "status": "Data reloaded successfully."
    }
```

### Reload Model
- **Endpoint:** `/reload_model`
- **Method:** POST
- **Description:** Initiates a process to reload or retrain the recommendation model. Useful after model updates or tuning.
- **Body:** None required for simple reloads.
- **Example Request**:
```
    POST /reload_model
```
- **Example Response**:
```json
    {
        "status": "Model reloaded successfully."
    }
```

## Model Selection and Challenges

The project aims to build a robust recommendation system, exploring two models:
1. A Neural Network based on embedded item and user information.
2. An Alternating Least Squares model using matrix factorization techniques.

Key challenges addressed:
- Dealing with unbalanced data through undersampling.
- Managing new or unknown users effectively.
- Optimizing model performance with respect to computational resources and maintainability.

The chosen model, a Neural Network, demonstrates better results in terms of AUC and adaptability for future improvements, such as integrating external data sources to enhance the recommendation quality.
