#!/bin/bash

# Colores para los mensajes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Función para mostrar mensajes
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Función para ejecutar pruebas
run_tests() {
    log "Ejecutando pruebas..."
    python -m pytest test_api.py -v
    if [ $? -eq 0 ]; then
        log "Todas las pruebas pasaron exitosamente"
        return 0
    else
        error "Las pruebas fallaron"
        return 1
    fi
}

# Función para iniciar los servicios
start_services() {
    log "Iniciando servicios..."
    docker-compose up -d zookeeper kafka
    sleep 10  # Esperar a que Kafka esté listo
    docker-compose up -d movie-recommender
}

# Función para detener los servicios
stop_services() {
    log "Deteniendo servicios..."
    docker-compose down
}

# Función para mostrar logs
show_logs() {
    log "Mostrando logs del servicio movie-recommender..."
    docker-compose logs -f movie-recommender
}

# Función para reconstruir el servicio
rebuild_service() {
    log "Reconstruyendo el servicio movie-recommender..."
    docker-compose build movie-recommender
    docker-compose up -d movie-recommender
}

# Menú principal
while true; do
    echo -e "\n${GREEN}=== Pipeline de Desarrollo ===${NC}"
    echo "1. Ejecutar pruebas"
    echo "2. Iniciar servicios"
    echo "3. Detener servicios"
    echo "4. Mostrar logs"
    echo "5. Reconstruir servicio"
    echo "6. Salir"
    echo -e "${GREEN}===========================${NC}"
    
    read -p "Seleccione una opción (1-6): " option
    
    case $option in
        1)
            run_tests
            ;;
        2)
            start_services
            ;;
        3)
            stop_services
            ;;
        4)
            show_logs
            ;;
        5)
            rebuild_service
            ;;
        6)
            log "Saliendo..."
            exit 0
            ;;
        *)
            error "Opción inválida"
            ;;
    esac
done 