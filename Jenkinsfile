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
                    # Detener todos los contenedores Docker
                    echo "Deteniendo todos los contenedores Docker..."
                    docker ps -q | xargs -r docker stop
                    docker ps -aq | xargs -r docker rm
                    
                    # Eliminar todos los volúmenes y redes no utilizados
                    docker system prune -f --volumes
                    
                    # Construir y desplegar nuevos contenedores
                    docker-compose up -d --build movie-recommender
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
                docker-compose down -v || true
                docker system prune -f --volumes || true
            '''
        }
    }
} 