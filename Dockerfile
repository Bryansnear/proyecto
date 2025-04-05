# Usar una imagen base de Python
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY api_service.py .
COPY recommender_model.py .
COPY kafka_service.py .
COPY data_fetcher.py .
COPY movies.csv .
COPY ratings.csv .
COPY recommender_model.pkl .
COPY static/ ./static/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 8082

# Comando para ejecutar la aplicación
CMD ["python", "api_service.py"] 