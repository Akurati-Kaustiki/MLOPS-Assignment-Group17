FROM python:3.10

# Set working directory
WORKDIR /app

# Copy everything including mlruns/ to /app/
COPY . .

# Set MLflow to use local mlruns directory inside container
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
