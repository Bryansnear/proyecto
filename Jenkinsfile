pipeline {
    agent any
    
    environment {
        PYTHON_PATH = 'C:/Users/bryan/AppData/Local/Programs/Python/Python312/python.exe'
        PIP_PATH = 'C:/Users/bryan/AppData/Local/Programs/Python/Python312/Scripts/pip.exe'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Environment Setup') {
            steps {
                bat """
                    "${PYTHON_PATH}" -m venv venv
                    call venv\\Scripts\\activate.bat
                    "${PIP_PATH}" install -r requirements.txt
                """
                bat """
                    echo Verificando conexion a Kafka...
                    netstat -an | findstr "9092"
                """
            }
        }
        
        stage('Test') {
            steps {
                bat """
                    call venv\\Scripts\\activate.bat
                    pytest test_api.py -v
                """
            }
        }
        
        stage('Build Docker') {
            steps {
                bat 'docker build -t movie-recommender:latest .'
            }
        }
        
        stage('Deploy') {
            steps {
                bat '''
                    docker ps -q --filter "name=movie-recommender" | findstr "." && docker stop movie-recommender || exit /b 0
                '''
                bat '''
                    docker run -d --name movie-recommender ^
                        -p 8082:8082 ^
                        -e KAFKA_HOST=host.docker.internal ^
                        -e KAFKA_PORT=9092 ^
                        movie-recommender:latest
                '''
            }
        }
        
        stage('Verify Deployment') {
            steps {
                bat '''
                    echo Esperando que la aplicacion este disponible...
                    timeout /t 10 /nobreak
                    curl -f http://localhost:8082/
                '''
            }
        }
    }
    
    post {
        always {
            bat 'rmdir /s /q venv'
            echo 'El pipeline ha fallado'
            bat '''
                docker ps -q --filter "name=movie-recommender" | findstr "." && docker stop movie-recommender || exit /b 0
            '''
        }
    }
} 