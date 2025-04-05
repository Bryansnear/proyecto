pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'movie-recommender'
        DOCKER_TAG = 'latest'
        GITHUB_REPO = 'https://github.com/Bryansnear/proyecto.git'
    }
    
    stages {
        stage('Checkout') {
            steps {
                // Obtener código de GitHub
                git branch: 'master',
                    url: env.GITHUB_REPO
            }
        }

        stage('Build') {
            steps {
                // Crear entorno virtual e instalar dependencias
                sh '''
                python -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }
        
        stage('Test') {
            steps {
                // Ejecutar pruebas
                sh '''
                . venv/bin/activate
                pytest test_api.py -v
                '''
            }
        }
        
        stage('Build and Deploy Docker') {
            steps {
                // Construir imagen Docker
                sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                
                // Detener contenedor existente si existe
                sh '''
                docker ps -q --filter "name=movie-recommender" | grep -q . && docker stop movie-recommender || true
                docker ps -aq --filter "name=movie-recommender" | grep -q . && docker rm movie-recommender || true
                '''
                
                // Ejecutar nuevo contenedor
                sh "docker run -d --name movie-recommender -p 8082:8082 ${DOCKER_IMAGE}:${DOCKER_TAG}"
            }
        }
    }
    
    post {
        always {
            // Limpiar entorno virtual
            sh 'rm -rf venv'
        }
        success {
            echo 'Pipeline ejecutado exitosamente!'
        }
        failure {
            echo 'El pipeline ha fallado'
        }
    }
} 