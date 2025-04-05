pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'movie-recommender'
        DOCKER_TAG = 'latest'
        KAFKA_HOST = 'localhost'
        KAFKA_PORT = '9092'
        PYTHON_PATH = '"C:/Users/bryan/AppData/Local/Programs/Python/Python312/python.exe"'
        PIP_PATH = '"C:/Users/bryan/AppData/Local/Programs/Python/Python312/Scripts/pip.exe"'
    }
    
    stages {
        stage('Checkout') {
            steps {
                // Obtener código de GitHub
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                // Crear entorno virtual e instalar dependencias
                bat """
                ${PYTHON_PATH} -m venv venv
                call venv\\Scripts\\activate.bat
                ${PIP_PATH} install -r requirements.txt
                """
                
                // Verificar que Kafka está disponible
                bat """
                echo Verificando conexion a Kafka...
                netstat -an | findstr "${KAFKA_PORT}"
                """
            }
        }
        
        stage('Test') {
            steps {
                // Ejecutar pruebas
                bat """
                call venv\\Scripts\\activate.bat
                ${PIP_PATH} install pytest
                ${PYTHON_PATH} -m pytest test_api.py -v
                """
            }
        }
        
        stage('Build Docker') {
            steps {
                // Construir imagen Docker
                bat "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
            }
        }
        
        stage('Deploy') {
            steps {
                // Detener contenedor existente si existe
                bat """
                docker ps -q --filter "name=movie-recommender" | findstr "." && docker stop movie-recommender || exit /b 0
                docker ps -aq --filter "name=movie-recommender" | findstr "." && docker rm movie-recommender || exit /b 0
                """
                
                // Ejecutar nuevo contenedor con acceso a Kafka
                bat """
                docker run -d --name movie-recommender ^
                    -p 8082:8082 ^
                    --network host ^
                    -e KAFKA_HOST=${KAFKA_HOST} ^
                    -e KAFKA_PORT=${KAFKA_PORT} ^
                    ${DOCKER_IMAGE}:${DOCKER_TAG}
                """
            }
        }
        
        stage('Verify Deployment') {
            steps {
                // Esperar a que la aplicación esté lista
                bat """
                echo Esperando que la aplicacion este disponible...
                timeout /t 10 /nobreak
                curl -f http://localhost:8082/
                """
            }
        }
    }
    
    post {
        always {
            // Limpiar entorno virtual
            bat "rmdir /s /q venv"
        }
        success {
            echo 'Pipeline ejecutado exitosamente!'
        }
        failure {
            echo 'El pipeline ha fallado'
            // Detener el contenedor en caso de fallo
            bat """
            docker ps -q --filter "name=movie-recommender" | findstr "." && docker stop movie-recommender || exit /b 0
            """
        }
    }
} 