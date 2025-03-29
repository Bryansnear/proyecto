# main.py
import os
import argparse
import threading
import time
from recommender_model import train_model_from_csv
from api_service import app
from kafka_service import MovieKafkaConsumer
import data_fetcher

def fetch_movie_data(api_key):
    """Fetch movie data from OMDb API"""
    print("Fetching movie data from OMDb API...")
    search_terms = ['action', 'comedy', 'drama', 'sci-fi', 'thriller', 'horror', 'romance']
    data_fetcher.fetch_and_store_movies(api_key, search_terms)
    
def generate_ratings():
    """Generate synthetic ratings data"""
    print("Generating synthetic ratings data...")
    data_fetcher.generate_ratings_data()

def train_model():
    """Train the recommendation model"""
    print("Training recommendation model...")
    train_model_from_csv()

def run_api_service():
    """Run the Flask API service"""
    print("Starting API service on port 8082...")
    app.run(host='0.0.0.0', port=8082, threaded=True)

def run_kafka_consumer():
    """Run Kafka consumer to display logs"""
    consumer = MovieKafkaConsumer()
    consumer.consume_logs()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--fetch-data', action='store_true', help='Fetch data from OMDb API')
    parser.add_argument('--api-key', help='OMDb API key')
    parser.add_argument('--generate-ratings', action='store_true', help='Generate synthetic ratings')
    parser.add_argument('--train', action='store_true', help='Train recommendation model')
    parser.add_argument('--api', action='store_true', help='Run API service')
    parser.add_argument('--kafka-consumer', action='store_true', help='Run Kafka consumer')
    
    args = parser.parse_args()
    
    # Process command line arguments
    if args.fetch_data:
        api_key = args.api_key or os.environ.get('OMDB_API_KEY')
        if not api_key:
            print("Error: OMDb API key required. Use --api-key or set OMDB_API_KEY environment variable")
            exit(1)
        fetch_movie_data(api_key)
    
    if args.generate_ratings:
        generate_ratings()
    
    if args.train:
        train_model()
    
    if args.kafka_consumer:
        consumer_thread = threading.Thread(target=run_kafka_consumer)
        consumer_thread.daemon = True
        consumer_thread.start()
    
    if args.api:
        run_api_service()
    
    # If no arguments, run the complete pipeline
    if not (args.fetch_data or args.generate_ratings or args.train or args.api or args.kafka_consumer):
        print("Running complete pipeline...")
        
        # Check if movies.csv exists
        if not os.path.exists('movies.csv'):
            api_key = os.environ.get('OMDB_API_KEY')
            if not api_key:
                print("Error: Set the OMDB_API_KEY environment variable")
                exit(1)
            fetch_movie_data(api_key)
        
        # Check if ratings.csv exists
        if not os.path.exists('ratings.csv'):
            generate_ratings()
        
        # Train the model if it doesn't exist
        if not os.path.exists('recommender_model.pkl'):
            train_model()
        
        # Start Kafka consumer in a separate thread
        consumer_thread = threading.Thread(target=run_kafka_consumer)
        consumer_thread.daemon = True
        consumer_thread.start()
        
        # Run the API service in the main thread
        run_api_service()