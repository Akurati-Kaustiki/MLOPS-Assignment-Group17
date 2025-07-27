FROM python:3.10
WORKDIR /app
COPY . .
COPY mlruns/ mlruns/
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
