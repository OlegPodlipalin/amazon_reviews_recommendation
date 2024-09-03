import logging
from flask import Flask, request, jsonify
from classes.model import Model
from classes.app_data_processing import AppDataProcessor
from config import LOG_FILE


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AppLogger")
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler.setLevel(logging.DEBUG) 
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
console_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
adp = AppDataProcessor()
model = Model()


@app.route("/")
def home():
    return "Welcome to the Recommendation System API!"


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    user_ids = request.args.get("users")
    if not user_ids:
        return jsonify({"error": "No user IDs provided"}), 400
    
    users = user_ids.split(",")
    recommendations = {}
    for user in users:
        user = user.strip()
        if adp.known_user(user):
            user_data = adp.get_user_data(user)
            predictions = model.predict(user_data)
            recommendations[user] = adp.extract_recommendations(predictions)    
        else:
            recommendations[user] = adp.cold_start_recommendations
    return jsonify(recommendations)


@app.route("/random_users", methods=["GET"])
def get_random_users():
    n = request.args.get("n", default=1, type=int)
    return jsonify(adp.get_random_users(n))


@app.route("/reload", methods=["POST"])
def reload_data():
    adp.reload_clean_data()
    adp.reload_embeddings()
    return "Data reloaded successfully."


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
    