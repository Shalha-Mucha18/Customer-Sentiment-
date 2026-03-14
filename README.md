<div align="center">

# 🧠 Customer Sentiment Classification

**An Enterprise-Grade MLOps & NLP Deployment Pipeline**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-orange?logo=dvc)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS EKS](https://img.shields.io/badge/AWS-EKS_Deployed-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/eks/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This repository contains a full-stack **Machine Learning Operations (MLOps)** pipeline for Natural Language Processing (NLP). It demonstrates the complete lifecycle of putting a predictive model into a scalable production environment. 

Moving beyond standard Jupyter Notebooks, this project emphasizes **reproducibility, observability, and scalability** by integrating industry-level toolchains, deploying a predictive REST API to an Amazon EKS cluster, and actively monitoring model health with Prometheus and Grafana.



## 🌟 Business Value & Technical Highlights

* **Automated Data Pipelines (DVC + S3):** Replaces fragile, manual data handling with a declarative DAG, ensuring exact reproducibility of training datasets.
* **Continuous Model Tracking (MLflow + DagsHub):** Logs hyperparameters, metrics, and serialized artifacts automatically, maintaining a clean model registry for staged rollouts.
* **Real-time Inference API (Flask + Docker):** Provides an isolated, low-latency microservice capable of executing complex text-cleaning pipelines on the fly before running classifications.
* **High-Availability Cloud Deployment (Amazon EKS):** Utilizes Kubernetes to ensure the serving layer is elastic, highly available, and capable of handling significant traffic spikes.
* **Zero-Downtime CI/CD (GitHub Actions):** Enforces rigorous testing. Only validated code is packaged into an Amazon ECR image and pushed to the cluster.
* **Proactive System Observability (Prometheus + Grafana):** Independent telemetry servers track hardware performance and live API health, enabling rapid identification of deployment anomalies or data drift.

---

## 🛠️ System Architecture

Our tech stack represents the standard for modern, cloud-native ML applications:

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Data & Versioning** | DVC, Amazon S3 | Data provenance, pipeline DAG orchestration |
| **Model Tracking** | MLflow, DagsHub | Experiment hashing, artifact management |
| **Application Layer** | Flask, scikit-learn, NLTK | REST endpoint serving and NLP text processing |
| **Containerization** | Docker, Amazon ECR | Application environment isolation and image storage |
| **Orchestration** | Amazon EKS, Kubernetes | Distributed compute, load balancing, pod management |
| **Monitoring** | Prometheus, Grafana | Endpoint telemetry, latency scraping, UI dashboards |
| **CI/CD** | GitHub Actions | Automated builds, testing, and Docker push workflows |

---

## 📂 Project Structure

```text
customer-sentiment-classification/
├── .github/workflows/ci.yaml  # CI/CD pipeline targeting AWS ECR
├── data/                      # DVC-tracked dataset artifacts
├── flask_app/                 # Prediction serving microservice
│   ├── app.py                 # Core routing, `/predict` API & `/metrics`
│   └── requirements.txt       # Frozen Flask runtime dependencies
├── src/                       # Modulized ML Pipeline
│   ├── data_ingestion.py      # Automated data retrieval handlers
│   ├── data_preprocessing.py  # Cleansing and unicode normalization
│   ├── feature_engineering.py # NLP transformations (TF-IDF, Embeddings)
│   ├── model_building.py      # Algorithm training and evaluation constraints
│   ├── model_evaluation.py    # MLflow metric interception
│   ├── register_model.py      # DagsHub model promotion logic
│   └── logger/                # Standardized system logging across all stages
├── deployment.yaml            # Kubernetes (K8s) configuration for Pods & LoadBalancers
├── Dockerfile                 # Slim Docker image recipe for production deployment
├── dvc.yaml                   # Directed Acyclic Graph (DAG) for data pipelines
└── params.yaml                # Externally configurable hyperparameters
```

---

## 💻 Local Development Guide

### 1. Environment Setup

*Requires Conda and AWS CLI configured with active credentials.*

```bash
git clone https://github.com/your-username/customer-sentiment-classification.git
cd customer-sentiment-classification

conda create -n atlas python=3.10
conda activate atlas
pip install -r requirements.txt
```

### 2. DVC Data Synchronization
Ensure your AWS CLI possesses read access to the designated S3 bucket.
```bash
dvc pull
```

### 3. Pipeline Reproduction
To retrain the model locally using the full MLOps pipeline, inject your DagsHub token to authenticate MLflow tracking:
```bash
export CAPSTONE_TEST="<your-dagshub-token>"
dvc repro
```

---

## 🐳 Containerization & Local API Testing

Before pushing to EKS, validate the container builds correctly.

1. **Build the image:**
   ```bash
   docker build -t sentiment-api:latest .
   ```

2. **Execute the local server:**
   ```bash
   docker run -p 8888:5000 -e CAPSTONE_TEST="<your-dagshub-token>" sentiment-api:latest
   ```
   Navigate to `http://localhost:8888` to test the API locally.

---

## ☁️ Production Deployment (Amazon EKS)

The model serving layer is orchestrated via Amazon Elastic Kubernetes Service.

### 1. Provision Cluster (`eksctl`)
```bash
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name flask-app-nodes \
  --node-type t3.small \
  --nodes 1 --nodes-min 1 --nodes-max 1 --managed
```

### 2. Configure IAM Authorization
EC2 instances acting as worker nodes require IAM privileges to authenticate against ECR. Attach the `AmazonEC2ContainerRegistryReadOnly` policy to your new **Node IAM Role**.

### 3. Deploy Kubernetes Manifests
Apply the secret containing your DagsHub authentication, then initialize the deployment and service manifests.

```bash
# Inject credentials into the cluster securely
kubectl create secret generic capstone-secret \
    --from-literal=CAPSTONE_TEST=<your-dagshub-token>

# Apply the Deployment and Service configurations
kubectl apply -f deployment.yaml
```

Once the load balancer finishes provisioning, locate your external IP to route traffic:
```bash
kubectl get svc flask-app-service
```

---

## 📊 Observability & Telemetry

Advanced metric tracking runs independently from the core EKS cluster, guaranteeing that monitoring remains online even if the prediction nodes degrade.

- **Prometheus** runs on an independent EC2 instance (`t3.medium`, Port `9090`). It polls the Flask `/metrics` endpoint continuously.
- **Grafana** runs on an independent EC2 instance (`t3.medium`, Port `3000`), generating real-time dashboards mapping API latency, traffic load, and distribution of predicted sentiments.

---

## 🧹 Infrastructure Teardown

To prevent orphaned instances and unnecessary AWS billing, destroy resources using the following sequence:

```bash
kubectl delete -f deployment.yaml
kubectl delete secret capstone-secret

eksctl delete cluster --name flask-app-cluster --region us-east-1
```
></p>
