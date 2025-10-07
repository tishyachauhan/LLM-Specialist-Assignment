# ============================================================================
# DEPLOYMENT.md
# ============================================================================
# Deployment Guide

## Local Deployment

### Quick Start
```bash
# 1. Clone repository
git clone <repo-url>
cd rag-system

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Run
python main.py
```

### Docker Deployment
```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## Cloud Deployment

### AWS (EC2 + Docker)

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.medium (minimum)
   - Security Group: Open ports 80, 443, 8000

2. **Setup Server**
```bash
# SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone <repo-url>
cd rag-system
cp .env.example .env
# Edit .env with API keys
docker-compose -f docker-compose.prod.yml up -d
```

3. **Configure Domain (Optional)**
   - Point domain to EC2 IP
   - Setup SSL with Let's Encrypt

### GCP (Cloud Run)

1. **Build Container**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-system
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy rag-system \
  --image gcr.io/PROJECT_ID/rag-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars PINECONE_API_KEY=xxx,GEMINI_API_KEY=xxx
```

### Azure (Container Instances)

1. **Create Container Registry**
```bash
az acr create --resource-group myResourceGroup \
  --name ragSystemRegistry --sku Basic
```

2. **Build and Push**
```bash
az acr build --registry ragSystemRegistry \
  --image rag-system:latest .
```

3. **Deploy Container**
```bash
az container create \
  --resource-group myResourceGroup \
  --name rag-system \
  --image ragSystemRegistry.azurecr.io/rag-system:latest \
  --dns-name-label rag-system \
  --ports 8000 \
  --environment-variables \
    PINECONE_API_KEY=xxx \
    GEMINI_API_KEY=xxx
```

## Monitoring

### Health Checks
```bash
curl http://your-domain/api/v1/health
```

### Logs
```bash
# Docker logs
docker-compose logs -f

# Application logs
tail -f logs/app.log
```

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.prod.yml
deploy:
  replicas: 3
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Load Balancing
Use Nginx, AWS ALB, or GCP Load Balancer