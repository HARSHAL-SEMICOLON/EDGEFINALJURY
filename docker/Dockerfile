FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY ml/ ml/
COPY edge_service/ edge_service/
COPY dashboard/ dashboard/
# train inside the image so the model matches the container's scikit-learn version
RUN python ml/train_model.py && rm -f edge_service/edge_store.db dashboard/central.db
ENV PYTHONUNBUFFERED=1
EXPOSE 5000 8050
CMD ["python", "edge_service/app.py"]
