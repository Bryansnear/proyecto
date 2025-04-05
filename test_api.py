import pytest
from api_service import app
import json

@pytest.fixture
def client():
    """Configurar cliente de prueba"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Probar que la ruta principal responde"""
    response = client.get('/')
    assert response.status_code == 200

def test_evaluation_route(client):
    """Probar que la ruta de evaluación responde"""
    response = client.get('/evaluation.html')
    assert response.status_code == 200

def test_recommend_route(client):
    """Probar que la ruta de recomendaciones responde"""
    response = client.get('/recommend/1')
    assert response.status_code == 200
    # Verificar que devuelve una lista de 20 recomendaciones
    recommendations = response.data.decode().split(',')
    assert len(recommendations) == 20

def test_movie_details_route(client):
    """Probar que la ruta de detalles de película responde"""
    response = client.get('/movie/1')
    assert response.status_code == 200
    data = json.loads(response.data)
    # Verificar que contiene los campos necesarios
    assert 'title' in data
    assert 'genre' in data

def test_evaluate_route(client):
    """Probar que la ruta de evaluación de accuracy responde"""
    response = client.get('/evaluate')
    assert response.status_code == 200
    data = json.loads(response.data)
    # Verificar que contiene las métricas necesarias
    assert 'diagnostics' in data
    assert 'accuracy_metrics' in data['diagnostics']
    metrics = data['diagnostics']['accuracy_metrics']
    assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1', 'mae', 'mse']) 