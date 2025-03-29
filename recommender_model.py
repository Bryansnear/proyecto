# recommender_model.py
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors  # Model 1: KNN
from sklearn.decomposition import TruncatedSVD  # Model 2: SVD
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import pickle
import random

class MovieRecommender:
    def __init__(self):
        self.knn_model = None
        self.svd_model = None
        self.best_model = 'knn'  # Default model
        self.ratings_df = None
        self.movies_df = None
        self.user_matrix = None
    
    def load_data(self, movies_file='movies.csv', ratings_file='ratings.csv'):
        """Load datasets"""
        self.movies_df = pd.read_csv(movies_file)
        self.ratings_df = pd.read_csv(ratings_file)
        print(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
    
    def train(self):
        """Train both models and select the better one"""
        print("Preparing data...")
        # Create user-movie matrix
        self.user_matrix = self.ratings_df.pivot(
            index='user_id', columns='movie_id', values='rating'
        ).fillna(0)
        
        # Convert to sparse matrix
        matrix = csr_matrix(self.user_matrix.values)
        
        print("Training KNN model...")
        # Train KNN model
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
        self.knn_model.fit(matrix)
        
        print("Training SVD model...")
        # Train SVD model
        self.svd_model = TruncatedSVD(n_components=10, random_state=42)
        self.svd_model.fit(matrix)
        
        # Compare models and select the better one
        knn_score = self._evaluate_model('knn')
        svd_score = self._evaluate_model('svd')
        
        print(f"Model evaluation - KNN: {knn_score:.4f}, SVD: {svd_score:.4f}")
        
        # Lower error is better
        self.best_model = 'svd' if svd_score < knn_score else 'knn'
        print(f"Selected {self.best_model.upper()} as best model")
    
    def _evaluate_model(self, model_type):
        """Simple evaluation by hiding some ratings and trying to predict them"""
        # Create a copy of ratings and hide 10% for testing
        sample_size = min(100, len(self.ratings_df) // 10)
        test_ratings = self.ratings_df.sample(sample_size)
        
        errors = []
        for _, row in test_ratings.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            actual_rating = row['rating']
            
            try:
                if model_type == 'knn':
                    predicted = self._predict_rating_knn(user_id, movie_id)
                else:
                    predicted = self._predict_rating_svd(user_id, movie_id)
                
                if predicted:
                    errors.append((predicted - actual_rating) ** 2)
            except:
                pass
        
        return np.sqrt(np.mean(errors)) if errors else float('inf')
    
    def _predict_rating_knn(self, user_id, movie_id):
        """Predict a rating using KNN model"""
        if user_id not in self.user_matrix.index:
            return None
        
        # Get user's index
        user_idx = self.user_matrix.index.get_loc(user_id)
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(
            self.user_matrix.iloc[user_idx].values.reshape(1, -1)
        )
        
        # Get ratings from similar users
        similar_users = [self.user_matrix.index[idx] for idx in indices.flatten()]
        similar_ratings = self.ratings_df[
            (self.ratings_df['user_id'].isin(similar_users)) & 
            (self.ratings_df['movie_id'] == movie_id)
        ]
        
        # Return average rating
        if len(similar_ratings) > 0:
            return similar_ratings['rating'].mean()
        return None
    
    def _predict_rating_svd(self, user_id, movie_id):
        """Predict a rating using SVD model"""
        if user_id not in self.user_matrix.index:
            return None
        
        # Get user's index
        user_idx = self.user_matrix.index.get_loc(user_id)
        
        # Check if movie exists in matrix
        if movie_id not in self.user_matrix.columns:
            return None
            
        # Get movie's index
        movie_idx = self.user_matrix.columns.get_loc(movie_id)
        
        # Transform user vector to latent space
        user_vector = self.user_matrix.iloc[user_idx].values.reshape(1, -1)
        user_latent = self.svd_model.transform(user_vector)
        
        # Reconstruct ratings
        reconstructed = self.svd_model.inverse_transform(user_latent)
        
        # Return predicted rating
        return reconstructed[0, movie_idx]
    
    def get_recommendations(self, user_id, top_n=20):
        """Get movie recommendations using the best model"""
        try:
            user_id = int(user_id)
            
            # Choose the appropriate model
            if self.best_model == 'svd':
                return self._get_svd_recommendations(user_id, top_n)
            else:
                return self._get_knn_recommendations(user_id, top_n)
                
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return self._get_popular_movies(top_n)
    
    def _get_knn_recommendations(self, user_id, top_n=20):
        """Get recommendations using KNN model"""
        if user_id not in self.user_matrix.index:
            return self._get_popular_movies(top_n)
        
        # Get movies this user has already rated
        rated_movies = set(self.ratings_df[
            self.ratings_df['user_id'] == user_id
        ]['movie_id'].tolist())
        
        # Get user's index
        user_idx = self.user_matrix.index.get_loc(user_id)
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(
            self.user_matrix.iloc[user_idx].values.reshape(1, -1)
        )
        
        # Get similar users
        similar_users = [self.user_matrix.index[idx] for idx in indices.flatten()]
        
        # Get movies rated highly by similar users
        similar_ratings = self.ratings_df[
            (self.ratings_df['user_id'].isin(similar_users)) & 
            (~self.ratings_df['movie_id'].isin(rated_movies)) &
            (self.ratings_df['rating'] >= 4)  # Only highly rated movies
        ]
        
        # Group by movie and count recommendations
        if len(similar_ratings) > 0:
            movie_counts = similar_ratings.groupby('movie_id').size()
            recommended = movie_counts.sort_values(ascending=False).index.tolist()[:top_n]
        else:
            recommended = []
        
        # Fill with popular movies if needed
        if len(recommended) < top_n:
            popular = self._get_popular_movies(top_n)
            for movie in popular:
                if movie not in recommended and movie not in rated_movies:
                    recommended.append(movie)
                if len(recommended) >= top_n:
                    break
        
        return recommended[:top_n]
    
    def _get_svd_recommendations(self, user_id, top_n=20):
        """Get recommendations using SVD model"""
        if user_id not in self.user_matrix.index:
            return self._get_popular_movies(top_n)
        
        # Get movies this user has already rated
        rated_movies = set(self.ratings_df[
            self.ratings_df['user_id'] == user_id
        ]['movie_id'].tolist())
        
        # Get user's index
        user_idx = self.user_matrix.index.get_loc(user_id)
        
        # Transform user vector to latent space
        user_vector = self.user_matrix.iloc[user_idx].values.reshape(1, -1)
        user_latent = self.svd_model.transform(user_vector)
        
        # Reconstruct full ratings vector
        predicted_ratings = self.svd_model.inverse_transform(user_latent)[0]
        
        # Create (movie_id, predicted_rating) pairs
        movie_ratings = list(zip(self.user_matrix.columns, predicted_ratings))
        
        # Filter movies the user has already rated
        movie_ratings = [(movie, rating) for movie, rating in movie_ratings 
                        if movie not in rated_movies]
        
        # Sort by predicted rating
        movie_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N movie IDs
        recommended = [movie for movie, _ in movie_ratings[:top_n]]
        
        # Fill with popular movies if needed
        if len(recommended) < top_n:
            popular = self._get_popular_movies(top_n)
            for movie in popular:
                if movie not in recommended and movie not in rated_movies:
                    recommended.append(movie)
                if len(recommended) >= top_n:
                    break
        
        return recommended[:top_n]
    
    def _get_popular_movies(self, n=20):
        """Get popular movies based on average rating"""
        try:
            # Group by movie and calculate average rating
            movie_stats = self.ratings_df.groupby('movie_id')['rating'].mean()
            # Sort by average rating
            popular = movie_stats.sort_values(ascending=False)
            # Return top N movie IDs
            return popular.index.tolist()[:n]
        except:
            # Return some movie IDs as fallback
            return self.movies_df['movie_id'].head(n).tolist()
    
    def save_model(self, filepath='recommender_model.pkl'):
        """Save the trained model"""
        model_data = {
            'knn_model': self.knn_model,
            'svd_model': self.svd_model,
            'best_model': self.best_model,
            'user_matrix': self.user_matrix
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='recommender_model.pkl'):
        """Load the trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.knn_model = model_data['knn_model']
        self.svd_model = model_data['svd_model'] 
        self.best_model = model_data['best_model']
        self.user_matrix = model_data['user_matrix']
        print(f"Model loaded from {filepath}")

# Function to train and save model
def train_model_from_csv(movies_file='movies.csv', ratings_file='ratings.csv', save_to='recommender_model.pkl'):
    recommender = MovieRecommender()
    recommender.load_data(movies_file, ratings_file)
    recommender.train()
    recommender.save_model(save_to)
    return recommender