# api_service.py
from flask import Flask, send_from_directory
import time
import socket
import pandas as pd
from recommender_model import MovieRecommender
from kafka_service import MovieKafkaProducer
import os

app = Flask(__name__)

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Load model
model = MovieRecommender()
try:
    model.load_model('recommender_model.pkl')
    print(f"Model loaded successfully (using {model.best_model.upper()} model)")
except:
    print("No model found, will train when needed")

# Kafka producer for logging
kafka_producer = MovieKafkaProducer()
server_hostname = socket.gethostname()

@app.route('/', methods=['GET'])
def index():
    """Serve the HTML interface"""
    return send_from_directory('static', 'index.html')

@app.route('/recommend/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    """Return top 20 recommendations for a user"""
    start_time = time.time()
    
    try:
        # Ensure model is loaded
        if model.knn_model is None or model.svd_model is None:
            print("Training new model...")
            model.load_data('movies.csv', 'ratings.csv')
            model.train()
            model.save_model('recommender_model.pkl')
        
        # Get top 20 recommendations
        recommendations = model.get_recommendations(user_id, 20)
        
        # Create comma-separated list
        result = ','.join(map(str, recommendations))
        
        # Calculate response time
        response_time = int((time.time() - start_time) * 1000)
        
        # Log to Kafka
        kafka_producer.log_recommendation(
            user_id=user_id,
            server=server_hostname, 
            status=200,
            recommendations=result,
            response_time=f"{response_time}ms"
        )
        
        return result, 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        print(f"Error: {e}")
        response_time = int((time.time() - start_time) * 1000)
        
        # Fallback to sequential numbers as last resort
        fallback = ','.join(str(i) for i in range(1, 21))
        
        # Log error to Kafka
        kafka_producer.log_recommendation(
            user_id=user_id,
            server=server_hostname,
            status=500,
            recommendations="error",
            response_time=f"{response_time}ms"
        )
        
        return fallback, 200, {'Content-Type': 'text/plain'}

@app.route('/movie/<movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """Get movie details by ID"""
    try:
        # Load movies data
        movies_df = pd.read_csv('movies.csv')
        
        # Find the movie
        movie = movies_df[movies_df['movie_id'] == int(movie_id)]
        
        if len(movie) == 0:
            return {"error": "Movie not found"}, 404
        
        # Return first matching movie as JSON
        return movie.iloc[0].to_dict(), 200
    except Exception as e:
        print(f"Error getting movie details: {e}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)