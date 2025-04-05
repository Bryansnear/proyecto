pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    apt-get update
                    apt-get install -y python3 python3-pip python3-venv curl
                    
                    # Instalar docker-compose
                    curl -L "https://github.com/docker/compose/releases/download/v2.24.6/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
                    chmod +x /usr/local/bin/docker-compose
                '''
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    python -m pytest test_api.py -v
                '''
            }
        }
        
        stage('Build and Deploy') {
            steps {
                sh '''
                    # Mostrar directorio actual y contenido
                    pwd
                    ls -la
                    
                    # Verificar que docker-compose.yml existe
                    if [ ! -f "docker-compose.yml" ]; then
                        echo "Error: docker-compose.yml no encontrado"
                        exit 1
                    fi
                    
                    # Detener y eliminar contenedor existente si existe
                    if docker ps -a --format '{{.Names}}' | grep -q '^movie-recommender$'; then
                        echo "Deteniendo y eliminando contenedor existente..."
                        docker stop movie-recommender
                        docker rm movie-recommender
                    fi
                    
                    # Construir y desplegar nuevo contenedor
                    docker-compose up -d --build movie-recommender
                '''
            }
        }
        
        stage('Verify Deployment') {
            steps {
                sh '''
                    echo "Esperando que la aplicacion este disponible..."
                    sleep 10
                    curl -f http://localhost:8082/
                '''
            }
        }
    }
    
    post {
        always {
            sh '''
                rm -rf venv
                if docker ps -a --format '{{.Names}}' | grep -q '^movie-recommender$'; then
                    echo "Limpiando contenedor..."
                    docker stop movie-recommender || true
                    docker rm movie-recommender || true
                fi
            '''
        }
    }
} 