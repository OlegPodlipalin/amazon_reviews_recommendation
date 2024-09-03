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
    """Create and return help message for the Flask application"""
    return """
    <pre>
    Welcome to the Recommendation System API!

    This API provides a set of endpoints for interacting with the recommendation engine.
    Below you will find the available REST endpoints along with examples on how to use them.

    1. Get Random Users
        Endpoint: /random_users
        Method: GET
        Description: Retrieves one or more random users from the system.
        Parameters:
            n (optional): Specifies the number of random users to retrieve.
        Example Request:
            GET /random_users?n=5
        Example Response:
            {
                ["user1", "user2", "user3", "user4", "user5"]
            }
    2. Check User Existence
        Endpoint: /check_users
        Method: GET
        Description: Checks if specified users are known in the system.
        Parameters:
            users: A comma-separated list of users to check.
        Example Request:
            GET /check_users?users=user1,user2
        Example Response:
            {
                "user1": True,
                "user2": False
            }
    3. Get Personalized Recommendations
        Endpoint: /recommendations
        Method: GET
        Description: Provides personalized product or content recommendations for specified users.
        Parameters:
            users: A comma-separated list of users to get recommendations for.
        Example Request:
            GET /recommendations?users=user1,user2
        Example Response:
            {
                "user1": ["product1", "product2"],
                "user2": ["product3", "product4"]
            }
    4. Reload Data
        Endpoint: /reload_data
        Method: POST
        Description: Triggers a reload of the underlying data used by the recommendation system. This might be necessary after updating data files or databases.
        Body: None required for simple reloads. 
        Example Request:
            POST /reload_data
        Example Response:
            {
                "status": "Data reloaded successfully."
            }
    5. Reload Model
        Endpoint: /reload_model
        Method: POST
        Description: Initiates a process to reload or retrain the recommendation model. Useful after model updates or tuning.
        Body: None required for simple reloads.
        Example Request:
            POST /reload_model
        Example Response:
            {
                "status": "Model reloaded successfully."
            }

    Notes:
        All responses are returned in JSON format.
    Ensure requests adhere to the correct content-type (application/json) where necessary.
    </pre>
    """


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    """Get list of personalized recommendations for each user
    Endpoint: /recommendations
    Method: GET
    Description: Provides personalized product or content recommendations for specified users.
    Parameters:
        users: A comma-separated list of users to get recommendations for.
    Example Request:
        GET /recommendations?users=user1,user2
    Example Response:
        {
            "user1": ["product1", "product2"],
            "user2": ["product3", "product4"]
        }
    """
    user_ids = request.args.get("users")
    if not user_ids:
        logger.error("no users provided with the recommendations request")
        return jsonify({"error": "No users provided"}), 400
    
    users = user_ids.split(",")
    logger.info(f"request for personalized recommendations for users: {users}")
    recommendations = {}
    for user in users:
        user = user.strip()
        if adp.known_user(user):
            user_data = adp.get_user_data(user)
            predictions = model.predict(user_data)
            recommendations[user] = adp.extract_recommendations(predictions)    
        else:
            logger.debug(f"cold start recommendation will be used for user: {user}")
            recommendations[user] = adp.cold_start_recommendations
    logger.info(f"recommendations for users: {recommendations}")
    return jsonify(recommendations)


@app.route("/random_users", methods=["GET"])
def get_random_users():
    """Get list of random known users
    Endpoint: /random_users
    Method: GET
    Description: Retrieves one or more random users from the system.
    Parameters:
        n (optional): Specifies the number of random users to retrieve.
    Example Request:
        GET /random_users?n=5
    Example Response:
        {
            ["user1", "user2", "user3", "user4", "user5"]
        }
    """
    n = request.args.get("n", default=1, type=int)
    logger.info(f"request for list of {n} random users")
    return jsonify(adp.get_random_users(n))


@app.route("/check_users", methods=["GET"])
def check_users():
    """Check user existence in a list of known users
    Endpoint: /check_users
    Method: GET
    Description: Checks if specified users are known in the system.
    Parameters:
        users: A comma-separated list of users to check.
    Example Request:
        GET /check_users?users=user1,user2
    Example Response:
        {
            "user1": True,
            "user2": False
        }
    """
    user_ids = request.args.get("users")
    if not user_ids:
        logger.error("no users provided this the check_users request")
        return jsonify({"error": "No user IDs provided"}), 400
    
    users = user_ids.split(",")
    logger.info(f"request to check users ({users}) in the list of known users")
    result = {}
    for user in users:
        result[user] = adp.known_user(user)
    logger.info(f"users status: {result}")
    return jsonify(result)


@app.route("/reload_data", methods=["POST"])
def reload_data():
    """Reload data
    Endpoint: /reload_data
    Method: POST
    Description: Triggers a reload of the underlying data used by the recommendation system. This might be necessary after updating data files or databases.
    Body: None required for simple reloads. 
    Example Request:
        POST /reload_data
    Example Response:
        {
            "status": "Data reloaded successfully."
        }
    """
    logger.warning(f"request for reloading data")
    adp.reload_clean_data()
    adp.reload_embeddings()
    return jsonify({"status": "Data reloaded successfully."})


@app.route("/reload_model", methods=["POST"])
def reload_model():
    """Reload model
    Endpoint: /reload_model
    Method: POST
    Description: Initiates a process to reload or retrain the recommendation model. Useful after model updates or tuning.
    Body: None required for simple reloads.
    Example Request:
        POST /reload_model
    Example Response:
        {
            "status": "Model reloaded successfully."
        }
    """
    logger.warning(f"request for reloading model")
    model.reload_model()
    return jsonify({"status": "Model reloaded successfully."})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
    