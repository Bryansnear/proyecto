pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    apt-get update
                    apt-get install -y python3 python3-pip python3-venv curl lsof findutils
                    
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
                    echo "Verificando contenedores existentes..."
                    CONTAINER_ID=$(docker ps -q --filter name=movie-recommender)
                    if [ ! -z "$CONTAINER_ID" ]; then
                        echo "Deteniendo contenedor movie-recommender existente (ID: $CONTAINER_ID)..."
                        docker stop $CONTAINER_ID
                        docker rm $CONTAINER_ID
                    else
                        echo "No se encontraron contenedores movie-recommender en ejecución"
                    fi
                    
                    echo "Limpiando recursos no utilizados..."
                    docker container prune -f
                    docker volume prune -f
                    
                    echo "Limpiando ambiente Docker Compose..."
                    docker-compose down -v
                    
                    echo "Limpiando redes Docker..."
                    docker network ls | grep movie-recommender | awk '{print $1}' | xargs -r docker network rm || true
                    
                    echo "Construyendo y desplegando nuevo contenedor..."
                    docker-compose build movie-recommender
                    echo "Iniciando contenedor..."
                    docker-compose up -d
                    
                    echo "Verificando estado del contenedor..."
                    docker ps --filter name=movie-recommender
                    
                    echo "Verificando logs del contenedor..."
                    sleep 5
                    docker logs movie-recommender-movie-recommender-1
                '''
            }
        }
        
        stage('Verify Deployment') {
            steps {
                sh '''
                    echo "Esperando que la aplicacion este disponible..."
                    echo "Verificando red Docker..."
                    docker network ls
                    
                    echo "Verificando estado de los contenedores..."
                    docker ps
                    
                    echo "Esperando 30 segundos para que el servicio se inicie completamente..."
                    sleep 30
                    
                    echo "Verificando logs del contenedor..."
                    docker logs movie-recommender-movie-recommender-1
                    
                    echo "Intentando acceder al servicio..."
                    curl -v --retry 3 --retry-delay 5 http://localhost:8082/
                '''
            }
        }
    }
    
    post {
        always {
            sh '''
                rm -rf venv
                if docker ps -q --filter name=movie-recommender; then
                    docker-compose stop movie-recommender || true
                    docker-compose rm -f movie-recommender || true
                fi
            '''
        }
    }
} 