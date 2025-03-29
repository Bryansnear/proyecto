# data_fetcher.py
import requests
import csv
import os
import time
import pandas as pd
import numpy as np

class OMDbClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://www.omdbapi.com/"
    
    def get_movie_details(self, movie_id):
        params = {'apikey': self.api_key, 'i': movie_id}
        response = requests.get(self.base_url, params=params)
        return response.json() if response.status_code == 200 else None
    
    def search_movies(self, query):
        params = {'apikey': self.api_key, 's': query, 'type': 'movie'}
        response = requests.get(self.base_url, params=params)
        return response.json() if response.status_code == 200 else None

def fetch_and_store_movies(api_key, search_terms, output_file='movies.csv', max_movies=500):
    client = OMDbClient(api_key)
    movie_count = 0
    
    # Define CSV headers
    headers = ['movie_id', 'imdb_id', 'title', 'year', 'genre', 'director', 'actors', 'plot', 'imdb_rating']
    
    # Create or overwrite CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        print(f"Fetching movie data from OMDb API and writing to {output_file}...")
        
        # Track movies we've already added to avoid duplicates
        added_movies = set()
        
        for term in search_terms:
            if movie_count >= max_movies:
                break
                
            search_results = client.search_movies(term)
            if not search_results or search_results.get('Response') != 'True':
                continue
                
            for movie in search_results.get('Search', []):
                if movie_count >= max_movies:
                    break
                    
                imdb_id = movie.get('imdbID')
                
                # Skip if we've already added this movie
                if imdb_id in added_movies:
                    continue
                    
                # Get full movie details
                details = client.get_movie_details(imdb_id)
                if not details or details.get('Response') != 'True':
                    continue
                
                try:
                    # Assign a unique movie_id (required for the lab)
                    movie_id = str(movie_count + 1)
                    
                    # Add movie to CSV
                    writer.writerow({
                        'movie_id': movie_id,
                        'imdb_id': imdb_id,
                        'title': details.get('Title', ''),
                        'year': details.get('Year', ''),
                        'genre': details.get('Genre', ''),
                        'director': details.get('Director', ''),
                        'actors': details.get('Actors', ''),
                        'plot': details.get('Plot', ''),
                        'imdb_rating': details.get('imdbRating', '0')
                    })
                    
                    added_movies.add(imdb_id)
                    movie_count += 1
                    print(f"Added movie: {details.get('Title')} ({details.get('Year')}) [{movie_count}/{max_movies}]")
                except Exception as e:
                    print(f"Error adding movie {imdb_id}: {str(e)}")
                
                # Respect API rate limits
                time.sleep(0.2)
    
    print(f"CSV populated with {movie_count} movies")

# Generate a synthetic user-movie ratings dataset for training
def generate_ratings_data(movies_csv='movies.csv', output_file='ratings.csv', num_users=100, rating_density=0.3):
    """Generate synthetic user ratings data with diverse preferences"""
    # Read movies data
    movies_df = pd.read_csv(movies_csv)
    movie_ids = movies_df['movie_id'].tolist()
    
    # Import numpy directly
    import numpy as np
    
    # Create genre dictionary for each movie
    movie_genres = {}
    for _, movie in movies_df.iterrows():
        if pd.notna(movie['genre']):
            genres = str(movie['genre']).lower().split(',')
            movie_genres[str(movie['movie_id'])] = [g.strip() for g in genres]
        else:
            movie_genres[str(movie['movie_id'])] = []
    
    # Create distinct user profiles (each likes different genres)
    all_genres = ['action', 'comedy', 'drama', 'thriller', 'sci-fi', 'horror', 'romance', 'adventure']
    user_profiles = []
    
    for i in range(num_users):
        # Each user likes 1-4 genres
        num_preferred_genres = np.random.randint(1, 5)
        if len(all_genres) > 0:  # Make sure there are genres to choose from
            preferred = np.random.choice(all_genres, size=min(num_preferred_genres, len(all_genres)), replace=False)
            user_profiles.append({
                'user_id': i + 1,
                'preferred_genres': list(preferred),
                # Also add a base rating tendency (some users rate higher/lower on average)
                'rating_bias': np.random.uniform(-0.5, 0.5)
            })
    
    # Create ratings CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'movie_id', 'rating'])
        
        # Generate ratings for each user based on their profile
        for profile in user_profiles:
            user_id = profile['user_id']
            preferred_genres = profile['preferred_genres']
            rating_bias = profile['rating_bias']
            
            # Determine how many movies this user will rate
            num_to_rate = int(len(movie_ids) * rating_density)
            movies_to_rate = np.random.choice(movie_ids, size=num_to_rate, replace=False)
            
            for movie_id in movies_to_rate:
                # Check if the movie matches the user's preferred genres
                genres = movie_genres.get(movie_id, [])
                genre_match = any(g in preferred_genres for g in genres)
                
                # Generate a rating based on genre match and bias
                base_rating = np.random.randint(1, 6)
                if genre_match:
                    rating = min(max(base_rating + rating_bias, 1), 5)  # Ensure rating is between 1 and 5
                else:
                    rating = base_rating
                
                writer.writerow([user_id, movie_id, round(rating, 1)])
    
    print(f"Generated ratings data for {num_users} users with {rating_density*100}% density")
    return output_file

if __name__ == "__main__":
    api_key = '7d4cfe12'
    if not api_key:
        print("Error: Set the OMDB_API_KEY environment variable")
        exit(1)
    
    search_terms = ['action', 'comedy', 'drama', 'sci-fi', 'thriller', 'horror', 'romance']
    fetch_and_store_movies(api_key, search_terms)
    generate_ratings_data()