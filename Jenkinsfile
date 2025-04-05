pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    apt-get update
                    apt-get install -y python3 python3-pip python3-venv curl lsof
                    
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
                    # Detener y eliminar todos los contenedores existentes
                    echo "Deteniendo y eliminando contenedores existentes..."
                    docker-compose down -v || true
                    
                    # Asegurarse de que los puertos estén libres
                    for port in 8081 8082 9093; do
                        pid=$(lsof -ti :$port || true)
                        if [ ! -z "$pid" ]; then
                            echo "Liberando puerto $port..."
                            kill -9 $pid || true
                        fi
                    done
                    
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
            '''
        }
    }
} 