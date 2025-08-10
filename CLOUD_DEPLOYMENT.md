# 🚀 Cloud Deployment Guide for RAG Chatbot

## 📋 Table of Contents
1. [Docker Hub Deployment](#docker-hub)
2. [AWS ECS Deployment](#aws-ecs)
3. [Google Cloud Run](#google-cloud-run)
4. [Azure Container Instances](#azure-container-instances)
5. [DigitalOcean App Platform](#digitalocean)
6. [Heroku Container Deployment](#heroku)

---

## 🐳 Docker Hub Deployment

### Step 1: Build and Push to Docker Hub
```bash
# Build the image
docker build -t yourusername/rag-chatbot:latest .

# Push to Docker Hub
docker login
docker push yourusername/rag-chatbot:latest
```

### Step 2: Run on Any Cloud Provider
```bash
docker run -p 5000:5000 yourusername/rag-chatbot:latest
```

---

## ☁️ AWS ECS Deployment

### Step 1: Create Task Definition
```json
{
  "family": "rag-chatbot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "rag-chatbot",
      "image": "yourusername/rag-chatbot:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-chatbot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Step 2: Deploy with AWS CLI
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name rag-chatbot-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster rag-chatbot-cluster \
  --service-name rag-chatbot-service \
  --task-definition rag-chatbot \
  --desired-count 1 \
  --launch-type FARGATE
```

---

## 🌐 Google Cloud Run

### Step 1: Deploy to Cloud Run
```bash
# Build and deploy in one command
gcloud run deploy rag-chatbot \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10
```

### Step 2: Get the deployment URL
```bash
gcloud run services describe rag-chatbot --region us-central1 --format 'value(status.url)'
```

---

## 💙 Azure Container Instances

### Step 1: Create Resource Group
```bash
az group create --name rag-chatbot-rg --location eastus
```

### Step 2: Deploy Container
```bash
az container create \
  --resource-group rag-chatbot-rg \
  --name rag-chatbot-container \
  --image yourusername/rag-chatbot:latest \
  --ports 5000 \
  --dns-name-label rag-chatbot-unique \
  --cpu 1 \
  --memory 1
```

---

## 🌊 DigitalOcean App Platform

### Step 1: Create app.yaml
```yaml
name: rag-chatbot
services:
- name: web
  source_dir: /
  github:
    repo: yourusername/rag-chatbot
    branch: master
  run_command: python flask_app/app.py
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  http_port: 5000
  routes:
  - path: /
```

### Step 2: Deploy with CLI
```bash
doctl apps create app.yaml
```

---

## 🟣 Heroku Container Deployment

### Step 1: Login and Create App
```bash
heroku login
heroku create your-rag-chatbot-app
heroku container:login
```

### Step 2: Build and Deploy
```bash
heroku container:push web --app your-rag-chatbot-app
heroku container:release web --app your-rag-chatbot-app
```

---

## 🛠️ Production Considerations

### Environment Variables
Set these in your cloud provider:
- `GOOGLE_API_KEY`: Your Google AI API key
- `SECRET_KEY`: Flask secret key
- `FLASK_ENV`: production

### Scaling Configuration
- **CPU**: 1-2 vCPUs for moderate traffic
- **Memory**: 1-2 GB RAM for embeddings
- **Storage**: Persistent volume for ChromaDB
- **Auto-scaling**: Based on CPU/memory usage

### Monitoring Setup
- Health checks on `/` endpoint
- Log aggregation for debugging
- Performance metrics monitoring
- Error alerting

### Security Best Practices
- Use HTTPS in production
- Implement rate limiting
- Validate file uploads
- Use secrets management
- Regular security updates

---

## 💰 Cost Estimates (Monthly)

| Provider | Basic Tier | Cost Range |
|----------|------------|------------|
| Google Cloud Run | 1 vCPU, 1GB RAM | $5-15 |
| AWS ECS Fargate | 0.25 vCPU, 0.5GB | $8-20 |
| Azure Container | 1 vCPU, 1GB RAM | $10-25 |
| DigitalOcean | Basic droplet | $12-24 |
| Heroku | Standard dyno | $25-50 |

*Costs vary based on usage and region*

---

## 📞 Interview Talking Points

When discussing your deployment:

1. **"I containerized my RAG application using Docker with multi-stage builds for optimization"**
2. **"Deployed on [Cloud Provider] with auto-scaling and health monitoring"**
3. **"Implemented production best practices: HTTPS, environment variables, logging"**
4. **"Used nginx as reverse proxy for load balancing and static file serving"**
5. **"Set up CI/CD pipeline for automated deployments"**

## 🎯 Next Steps
1. Choose your preferred cloud provider
2. Set up CI/CD pipeline with GitHub Actions
3. Implement monitoring and alerting
4. Add SSL certificates
5. Set up custom domain
