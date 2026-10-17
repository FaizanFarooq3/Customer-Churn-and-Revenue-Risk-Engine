#python base image 
FROM python:3.13-slim

#to set the working dir inside container
WORKDIR /app

#o install system dependencies required for building python packages (like psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

#copy requirements and install them
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

#copy the application source code
COPY src/ /app/src/
COPY model_export/ /app/model_export/

#setting python path so python knows where to find src module
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model_export

#expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API using Uvicorn
CMD ["uvicorn","src.api.main:app", "--host","0.0.0.0","--port","8000"]