# api_service.py
from flask import Flask, send_from_directory, jsonify
import time
import socket
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    # Primero cargar los datos
    model.load_data('movies.csv', 'ratings.csv')
    # Luego cargar el modelo entrenado
    model.load_model('recommender_model.pkl')
    print(f"Model loaded successfully (using {model.best_model.upper()} model)")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Will train new model when needed")
    # Si hay error, intentar cargar datos y entrenar
    try:
        model.load_data('movies.csv', 'ratings.csv')
        model.train()
        model.save_model('recommender_model.pkl')
        print("New model trained and saved successfully")
    except Exception as e:
        print(f"Error training new model: {e}")

# Kafka producer for logging
kafka_producer = MovieKafkaProducer()
server_hostname = socket.gethostname()

@app.route('/', methods=['GET'])
def index():
    """Serve the HTML interface"""
    return send_from_directory('static', 'index.html')

@app.route('/evaluation.html', methods=['GET'])
def evaluation():
    """Serve the evaluation HTML interface"""
    return send_from_directory('static', 'evaluation.html')

@app.route('/recommend/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    """Return top 20 recommendations for a user"""
    start_time = time.time()
    
    try:

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

@app.route('/evaluate', methods=['GET'])
def evaluate_accuracy():
    """Evalúa el accuracy del sistema de recomendaciones"""
    start_time = time.time()
    
    try:
        # Cargar ratings reales y películas
        ratings_df = pd.read_csv('ratings.csv')
        movies_df = pd.read_csv('movies.csv')
        
        # Obtener lista única de usuarios y convertir a lista nativa de Python
        unique_users = ratings_df['user_id'].unique().tolist()
        
        # Seleccionar 100 usuarios aleatorios o menos si no hay suficientes
        n_users = min(100, len(unique_users))
        selected_users = np.random.choice(unique_users, size=n_users, replace=False).tolist()
        
        # Estadísticas para diagnóstico
        stats = {
            'total_users': len(selected_users),
            'users_with_recommendations': 0,
            'users_with_matches': 0,
            'total_matches': 0,
            'sample_recommendations': [],
            'sample_ratings': [],
            'movie_coverage': {
                'total_movies': len(movies_df),
                'recommended_movies': set(),
                'rated_movies': set(ratings_df['movie_id'].unique().tolist())
            },
            'recommendation_stats': {
                'avg_rating': 0,
                'rating_distribution': {},
                'genre_distribution': {}
            },
            'accuracy_metrics': {
                'mse': 0,
                'mae': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
        }
        
        total_recommendations = 0
        total_ratings = 0
        all_predictions = []
        all_actuals = []
        binary_predictions = []
        binary_actuals = []
        
        for user_id in selected_users:
            try:
                # Obtener ratings reales del usuario
                user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                
                # Para cada película calificada por el usuario
                for _, row in user_ratings.iterrows():
                    movie_id = row['movie_id']
                    actual_rating = float(row['rating'])
                    
                    # Obtener predicción del modelo
                    predicted_rating = model._predict_rating_knn(user_id, movie_id) if model.best_model == 'knn' else model._predict_rating_svd(user_id, movie_id)
                    
                    if predicted_rating is not None:
                        all_predictions.append(predicted_rating)
                        all_actuals.append(actual_rating)
                        
                        # Para métricas binarias (recomendado vs no recomendado)
                        binary_predictions.append(1 if predicted_rating >= 4 else 0)
                        binary_actuals.append(1 if actual_rating >= 4 else 0)
                
                # Obtener recomendaciones para mostrar ejemplos
                recommendations = model.get_recommendations(user_id, 20)
                
                if recommendations is None:
                    continue
                    
                stats['users_with_recommendations'] += 1
                stats['movie_coverage']['recommended_movies'].update([int(x) for x in recommendations])
                
                # Obtener ratings promedio de las películas recomendadas
                movie_ratings = ratings_df[ratings_df['movie_id'].isin(recommendations)]
                if not movie_ratings.empty:
                    avg_rating = movie_ratings['rating'].mean()
                    total_ratings += avg_rating
                    total_recommendations += 1
                    
                    # Actualizar distribución de ratings
                    for rating in movie_ratings['rating']:
                        rating_key = str(int(rating))
                        stats['recommendation_stats']['rating_distribution'][rating_key] = \
                            stats['recommendation_stats']['rating_distribution'].get(rating_key, 0) + 1
                
                # Guardar una muestra de recomendaciones con títulos y ratings promedio
                if len(stats['sample_recommendations']) < 50:
                    recommended_movies_info = []
                    for movie_id in recommendations:
                        movie_info = movies_df[movies_df['movie_id'] == movie_id]
                        if not movie_info.empty:
                            # Obtener rating promedio de la película
                            movie_avg_rating = ratings_df[ratings_df['movie_id'] == movie_id]['rating'].mean()
                            # Obtener predicción del modelo
                            predicted_rating = model._predict_rating_knn(user_id, movie_id) if model.best_model == 'knn' else model._predict_rating_svd(user_id, movie_id)
                            recommended_movies_info.append({
                                'id': int(movie_id),
                                'title': str(movie_info.iloc[0]['title']),
                                'avg_rating': float(movie_avg_rating) if not pd.isna(movie_avg_rating) else 0.0,
                                'predicted_rating': float(predicted_rating) if predicted_rating is not None else 0.0,
                                'genre': str(movie_info.iloc[0]['genre'])
                            })
                    
                    stats['sample_recommendations'].append({
                        'user_id': int(user_id),
                        'recommendations': recommended_movies_info
                    })
                
                # Guardar una muestra de ratings con títulos
                if len(stats['sample_ratings']) < 50:
                    ratings_info = []
                    for _, row in user_ratings.iterrows():
                        movie_info = movies_df[movies_df['movie_id'] == row['movie_id']]
                        if not movie_info.empty:
                            # Obtener predicción del modelo
                            predicted_rating = model._predict_rating_knn(user_id, row['movie_id']) if model.best_model == 'knn' else model._predict_rating_svd(user_id, row['movie_id'])
                            ratings_info.append({
                                'movie_id': int(row['movie_id']),
                                'title': str(movie_info.iloc[0]['title']),
                                'rating': float(row['rating']),
                                'predicted_rating': float(predicted_rating) if predicted_rating is not None else 0.0,
                                'genre': str(movie_info.iloc[0]['genre'])
                            })
                    
                    stats['sample_ratings'].append({
                        'user_id': int(user_id),
                        'ratings': ratings_info
                    })
            except Exception as e:
                print(f"Error procesando usuario {user_id}: {e}")
                continue
        
        # Calcular métricas de accuracy
        if all_predictions and all_actuals:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
            
            stats['accuracy_metrics']['mse'] = float(mean_squared_error(all_actuals, all_predictions))
            stats['accuracy_metrics']['mae'] = float(mean_absolute_error(all_actuals, all_predictions))
            stats['accuracy_metrics']['accuracy'] = float(accuracy_score(binary_actuals, binary_predictions))
            stats['accuracy_metrics']['precision'] = float(precision_score(binary_actuals, binary_predictions, zero_division=0))
            stats['accuracy_metrics']['recall'] = float(recall_score(binary_actuals, binary_predictions, zero_division=0))
            stats['accuracy_metrics']['f1'] = float(f1_score(binary_actuals, binary_predictions, zero_division=0))
        
        # Calcular estadísticas finales
        if total_recommendations > 0:
            stats['recommendation_stats']['avg_rating'] = float(total_ratings / total_recommendations)
        
        # Convertir sets a listas para JSON
        stats['movie_coverage']['recommended_movies'] = list(stats['movie_coverage']['recommended_movies'])
        stats['movie_coverage']['rated_movies'] = list(stats['movie_coverage']['rated_movies'])
        
        # Calcular tiempo de respuesta
        response_time = int((time.time() - start_time) * 1000)
        
        # Log a Kafka
        kafka_producer.log_recommendation(
            user_id="evaluation",
            server=server_hostname,
            status=200,
            recommendations=f"Evaluación completada: {stats['users_with_recommendations']} usuarios, Accuracy: {stats['accuracy_metrics']['accuracy']:.4f}",
            response_time=f"{response_time}ms"
        )
        
        return jsonify({
            'status': 'success',
            'response_time_ms': response_time,
            'users_evaluated': stats['users_with_recommendations'],
            'total_users_attempted': len(selected_users),
            'diagnostics': stats
        })
        
    except Exception as e:
        print(f"Error en evaluación: {e}")
        response_time = int((time.time() - start_time) * 1000)
        
        # Log error a Kafka
        kafka_producer.log_recommendation(
            user_id="evaluation",
            server=server_hostname,
            status=500,
            recommendations="error",
            response_time=f"{response_time}ms"
        )
        
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)