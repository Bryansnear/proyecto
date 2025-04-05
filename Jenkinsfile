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
                    if docker ps -q --filter name=movie-recommender; then
                        echo "Deteniendo contenedor movie-recommender existente..."
                        docker stop $(docker ps -q --filter name=movie-recommender)
                        docker rm $(docker ps -aq --filter name=movie-recommender)
                    else
                        echo "No se encontraron contenedores movie-recommender en ejecución"
                    fi
                    
                    echo "Limpiando recursos no utilizados..."
                    docker container prune -f
                    docker volume prune -f
                    
                    echo "Construyendo y desplegando nuevo contenedor..."
                    docker-compose build movie-recommender
                    echo "Iniciando contenedor..."
                    docker-compose up -d movie-recommender
                    
                    echo "Verificando estado del contenedor..."
                    docker ps --filter name=movie-recommender
                '''
            }
        }
        
        stage('Verify Deployment') {
            steps {
                sh '''
                    echo "Esperando que la aplicacion este disponible..."
                    sleep 20
                    curl -f http://localhost:8082/
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