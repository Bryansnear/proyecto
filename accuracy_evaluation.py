import requests
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from tqdm import tqdm

def get_recommendations(user_id):
    """Obtiene recomendaciones del servicio para un usuario específico"""
    try:
        response = requests.get(f'http://localhost:8082/recommend/{user_id}')
        if response.status_code == 200:
            return [int(x) for x in response.text.split(',')]
        return None
    except Exception as e:
        print(f"Error al obtener recomendaciones para usuario {user_id}: {e}")
        return None

def evaluate_recommendations():
    # Cargar ratings reales
    ratings_df = pd.read_csv('ratings.csv')
    
    # Obtener lista única de usuarios
    unique_users = ratings_df['user_id'].unique()
    
    # Seleccionar 100 usuarios aleatorios
    selected_users = np.random.choice(unique_users, size=100, replace=False)
    
    results = {
        'user_id': [],
        'recommended_movies': [],
        'real_ratings': [],
        'mse': [],
        'mae': []
    }
    
    print("Evaluando recomendaciones para 100 usuarios...")
    for user_id in tqdm(selected_users):
        # Obtener recomendaciones del servicio
        recommended_movies = get_recommendations(user_id)
        
        if recommended_movies is None:
            continue
            
        # Obtener ratings reales del usuario
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        # Filtrar solo las películas que fueron recomendadas
        relevant_ratings = user_ratings[user_ratings['movie_id'].isin(recommended_movies)]
        
        if len(relevant_ratings) == 0:
            continue
            
        # Calcular métricas
        mse = mean_squared_error(relevant_ratings['rating'], [4.0] * len(relevant_ratings))  # Asumimos rating promedio de 4
        mae = mean_absolute_error(relevant_ratings['rating'], [4.0] * len(relevant_ratings))
        
        results['user_id'].append(user_id)
        results['recommended_movies'].append(recommended_movies)
        results['real_ratings'].append(relevant_ratings['rating'].tolist())
        results['mse'].append(mse)
        results['mae'].append(mae)
        
        # Pequeña pausa para no sobrecargar el servidor
        time.sleep(0.1)
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    
    # Calcular métricas globales
    print("\nResultados de la evaluación:")
    print(f"Promedio MSE: {results_df['mse'].mean():.4f}")
    print(f"Promedio MAE: {results_df['mae'].mean():.4f}")
    
    # Guardar resultados
    results_df.to_csv('accuracy_results.csv', index=False)
    print("\nResultados guardados en 'accuracy_results.csv'")

if __name__ == "__main__":
    evaluate_recommendations() 