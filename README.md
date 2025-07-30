# Iris Classification MLOps Pipeline

This repository demonstrates a **complete MLOps workflow** for classifying the Iris dataset using Logistic Regression, Random Forest, and SVM. It includes **data versioning, experiment tracking, API deployment, Docker containerization, CI/CD with GitHub Actions**, and basic **logging/monitoring**.

---

## Features

- Clean directory structure with Git tracking
- Train, track, and compare 3 models with MLflow
- Automatically log and register the best model
- FastAPI-based REST API for predictions
- Dockerized for consistent deployment
- Logging of predictions with timestamp
- Ready for CI/CD and Prometheus integration

---




# Iris Classification MLOps Pipeline

This repository demonstrates a **complete MLOps workflow** for classifying the Iris dataset using Logistic Regression, Random Forest, and SVM. It includes **data versioning, experiment tracking, API deployment, Docker containerization, CI/CD with GitHub Actions**, and basic **logging/monitoring**.

---

## Features

- Clean directory structure with Git tracking
- Train, track, and compare 3 models with MLflow
- Automatically log and register the best model
- FastAPI-based REST API for predictions
- Dockerized for consistent deployment
- Logging of predictions with timestamp
- Ready for CI/CD and Prometheus integration

---

## Project Structure

```
├── app/                  # FastAPI application
│   └── main.py
├── data/                 # Iris dataset
│   └── iris.csv
├── models/               # Trained best model (.pkl)
├── src/                  # Scripts for training and utils
│   ├── data/
│   │   └── preprocess.py
│   ├── training/
│   │   └── train.py
│   └── utils/
│       └── logger.py
├── logs/                 # API logs
├── mlruns/               # MLflow experiment tracking
├── requirements.txt
├── dvc.yaml
├── Dockerfile
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Akurati-Kaustiki/MLOPS-Assignment-Group17
cd iris-mlops
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
```bash
python src/data/preprocess.py
```

### 4. Train and log models with MLflow
```bash
python src/training/train.py
```

This:
- Trains Logistic Regression, Random Forest, and SVM
- Logs all models to MLflow
- Saves and registers the best model

---

## View MLflow UI
Open new terminal  -> Go to root directory of the project
```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

Visit:
- API root: [http://localhost:8000](http://localhost:8000)
- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker Instructions

### Build the image
```bash
docker build -t iris-api .
```

### Run the container
```bash
docker run -d -p 8000:8000 iris-api
```
Or 
### Run your container and mount the logs to your host (optional but recommended for access):
```bash
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs iris-api
```


---

## Sample API Request

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## CI/CD (GitHub Actions)

> Automatically:
- Lints the code
- Builds and pushes Docker image
- Deployed in local through Docker

```bash
docker run -p 8000:8000 akaustiki/iris-api:latest
docker run -p 8000:8000 -v \$(pwd)/logs:/app/logs akaustiki/iris-api:latest
```

Workflow file: `.github/workflows/main.yml`

---

## Logging & Monitoring

- All API predictions are logged to `logs/api.log`
- Storing the predictions to `logs/predictions.db`
- Easily extendable with `/metrics` endpoint and Prometheus

To Open logs/predictions.db
```bash
sqlite3 logs/predictions.db
.tables
select * from logs;
```

- Pometheus setup at `prometheus.yml`
1. Start Prometheus in Docker
```bash
docker run -d \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```
2. Access prometheus UI at http://localhost:9090
3. Run Grafana in Docker
```bash
docker run -d -p 3000:3000 --name grafana grafana/grafana
```
4. Access Grafana at http://localhost:3000 with default credentials:
    Username: admin
    Password: admin
5. Add Prometheus as a Data Source
    Go to ⚙️ Configuration > Data Sources
    Click Add data source
    Select Prometheus
    Set URL to: http://host.docker.internal:9090
    Click Save & Test
6. Build Your Dashboard
    Click ➕ Create > Dashboard > Add new panel
    Use queries such as:    
        http_requests_total – Requests by route
        http_requests_by_method_total – GET/POST breakdown
        http_request_duration_seconds_bucket – Latency histogram
    Customize visualization → Click Apply
    Save your dashboard with a name like Iris Model Monitoring

---

## Model Registry

Best model is automatically registered in **MLflow Model Registry** under:
```
Model Name: IrisClassifier
```

---

## Future Improvements
- DVC integration for large datasets
- CI/CD deployment to AWS or Render

---

## Contributing

Feel free to open issues or submit PRs. All contributions are welcome!

---

## License

MIT License © 2025
