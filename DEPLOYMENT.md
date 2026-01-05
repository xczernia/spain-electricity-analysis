# Deployment Guide

This document provides comprehensive deployment instructions for the Spain Electricity Analysis application across various platforms and environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker](#docker)
- [Heroku](#heroku)
- [AWS](#aws)
- [Google Cloud](#google-cloud)
- [Kubernetes](#kubernetes)

---

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv
- Git
- Database (SQLite for development, PostgreSQL for production)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/xczernia/spain-electricity-analysis.git
   cd spain-electricity-analysis
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the project root:
   ```
   FLASK_ENV=development
   FLASK_APP=app.py
   DATABASE_URL=sqlite:///app.db
   DEBUG=True
   SECRET_KEY=your-secret-key-here
   ```

5. **Initialize Database**
   ```bash
   python -c "from app import db; db.create_all()"
   ```

6. **Run the Application**
   ```bash
   flask run
   ```
   The application will be available at `http://localhost:5000`

### Development Notes

- The application uses SQLite by default for local development
- Hot reloading is enabled when `FLASK_ENV=development`
- API documentation available at `/api/docs`
- Database migrations use Alembic (if applicable)

---

## Docker

### Prerequisites

- Docker 20.10 or higher
- Docker Compose 1.29 or higher

### Using Docker Compose (Recommended)

1. **Create docker-compose.yml** (if not present)
   ```yaml
   version: '3.8'

   services:
     app:
       build: .
       container_name: spain-electricity-analysis
       ports:
         - "5000:5000"
       environment:
         - FLASK_ENV=production
         - DATABASE_URL=postgresql://user:password@db:5432/electricity_db
         - SECRET_KEY=${SECRET_KEY}
       depends_on:
         - db
       volumes:
         - ./data:/app/data
       networks:
         - electricity-network

     db:
       image: postgres:14-alpine
       container_name: electricity-db
       environment:
         - POSTGRES_USER=user
         - POSTGRES_PASSWORD=password
         - POSTGRES_DB=electricity_db
       volumes:
         - postgres_data:/var/lib/postgresql/data
       networks:
         - electricity-network

   volumes:
     postgres_data:

   networks:
     electricity-network:
       driver: bridge
   ```

2. **Build and Run**
   ```bash
   docker-compose up -d
   ```

3. **View Logs**
   ```bash
   docker-compose logs -f app
   ```

4. **Stop Services**
   ```bash
   docker-compose down
   ```

### Building Standalone Docker Image

1. **Build the Image**
   ```bash
   docker build -t spain-electricity-analysis:latest .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     -p 5000:5000 \
     -e DATABASE_URL="postgresql://user:password@db:5432/electricity_db" \
     -e SECRET_KEY="your-secret-key" \
     --name electricity-app \
     spain-electricity-analysis:latest
   ```

3. **Execute Commands in Container**
   ```bash
   docker exec -it electricity-app flask db upgrade
   ```

### Dockerfile Example

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "wsgi:app"]
```

---

## Heroku

### Prerequisites

- Heroku CLI installed
- Heroku account
- Git repository initialized

### Deployment Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create spain-electricity-analysis
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set FLASK_ENV=production
   heroku config:set SECRET_KEY="your-secure-secret-key"
   heroku config:set DATABASE_URL="postgresql://..."
   ```

4. **Add PostgreSQL Add-on**
   ```bash
   heroku addons:create heroku-postgresql:standard-0
   ```

5. **Create Procfile** (in project root)
   ```
   web: gunicorn wsgi:app
   worker: celery -A celery_app worker --loglevel=info
   ```

6. **Create runtime.txt** (specify Python version)
   ```
   python-3.9.16
   ```

7. **Deploy Application**
   ```bash
   git push heroku main
   ```

8. **Run Database Migrations**
   ```bash
   heroku run python -c "from app import db; db.create_all()"
   ```

9. **View Logs**
   ```bash
   heroku logs --tail
   ```

### Monitoring

- Dashboard: `heroku open`
- Metrics: `heroku metrics`
- Dyno scaling: `heroku ps:scale web=2`

### Scaling Configuration

```bash
# Scale web dynos
heroku ps:scale web=3

# Scale worker dynos
heroku ps:scale worker=2

# Check current processes
heroku ps
```

---

## AWS

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI v2 installed
- IAM user with EC2, RDS, and ECR access

### Deployment Options

### Option 1: EC2 + RDS

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS AMI
   - Instance type: t3.medium or larger
   - Configure security group to allow ports 80, 443, 22

2. **SSH into Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-instance-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3.9 python3-pip python3-venv nginx supervisor
   ```

4. **Clone Repository**
   ```bash
   cd /home/ubuntu
   git clone https://github.com/xczernia/spain-electricity-analysis.git
   cd spain-electricity-analysis
   ```

5. **Setup Application**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Configure Environment**
   ```bash
   cat > .env << EOF
   FLASK_ENV=production
   DATABASE_URL=postgresql://user:password@rds-endpoint:5432/electricity_db
   SECRET_KEY=your-secret-key
   EOF
   ```

7. **Setup Supervisor**
   Create `/etc/supervisor/conf.d/electricity.conf`:
   ```ini
   [program:electricity]
   directory=/home/ubuntu/spain-electricity-analysis
   command=/home/ubuntu/spain-electricity-analysis/venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 wsgi:app
   autostart=true
   autorestart=true
   user=ubuntu
   ```

8. **Configure Nginx**
   Create `/etc/nginx/sites-available/electricity`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

9. **Enable and Start Services**
   ```bash
   sudo systemctl restart supervisor
   sudo systemctl restart nginx
   ```

### Option 2: AWS ECS (Recommended for scalability)

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name spain-electricity-analysis --region us-east-1
   ```

2. **Build and Push Docker Image**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   
   docker build -t spain-electricity-analysis:latest .
   
   docker tag spain-electricity-analysis:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/spain-electricity-analysis:latest
   
   docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/spain-electricity-analysis:latest
   ```

3. **Create ECS Task Definition** (task-definition.json)
   ```json
   {
     "family": "spain-electricity",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "256",
     "memory": "512",
     "containerDefinitions": [
       {
         "name": "app",
         "image": "YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/spain-electricity-analysis:latest",
         "portMappings": [
           {
             "containerPort": 5000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "FLASK_ENV",
             "value": "production"
           }
         ],
         "secrets": [
           {
             "name": "DATABASE_URL",
             "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:electricity/db-url"
           }
         ]
       }
     ]
   }
   ```

4. **Register Task Definition**
   ```bash
   aws ecs register-task-definition --cli-input-json file://task-definition.json
   ```

5. **Create ECS Service** (via AWS Console or CLI)
   - Cluster: Create new or use existing
   - Launch type: Fargate
   - Task definition: spain-electricity:latest
   - Desired count: 2

6. **Setup RDS Database**
   ```bash
   aws rds create-db-instance \
     --db-instance-identifier electricity-db \
     --db-instance-class db.t3.micro \
     --engine postgres \
     --master-username admin \
     --master-user-password YOUR_PASSWORD \
     --allocated-storage 20
   ```

### Option 3: AWS Elastic Beanstalk

1. **Initialize Elastic Beanstalk**
   ```bash
   eb init -p python-3.9 spain-electricity-analysis
   ```

2. **Create Environment**
   ```bash
   eb create production-env
   ```

3. **Configure Environment Variables**
   ```bash
   eb setenv FLASK_ENV=production SECRET_KEY=your-key DATABASE_URL=postgresql://...
   ```

4. **Deploy Application**
   ```bash
   eb deploy
   ```

5. **Monitor Deployment**
   ```bash
   eb logs
   ```

---

## Google Cloud

### Prerequisites

- Google Cloud Project created
- gcloud CLI installed and configured
- Appropriate IAM permissions

### Deployment Option 1: Cloud Run (Serverless)

1. **Authenticate with GCP**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Build and Push Container Image**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/spain-electricity-analysis:latest
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy spain-electricity-analysis \
     --image gcr.io/YOUR_PROJECT_ID/spain-electricity-analysis:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars FLASK_ENV=production,SECRET_KEY=your-key
   ```

4. **Setup Cloud SQL**
   ```bash
   gcloud sql instances create electricity-db \
     --database-version POSTGRES_14 \
     --tier db-f1-micro \
     --region us-central1
   ```

5. **Create Database**
   ```bash
   gcloud sql databases create electricity_db \
     --instance electricity-db
   ```

6. **Set Cloud SQL Proxy Connection**
   Update the Cloud Run deployment to include Cloud SQL proxy:
   ```bash
   gcloud run deploy spain-electricity-analysis \
     --image gcr.io/YOUR_PROJECT_ID/spain-electricity-analysis:latest \
     --add-cloudsql-instances YOUR_PROJECT_ID:us-central1:electricity-db \
     --set-env-vars DATABASE_URL="postgresql://user:password@/electricity_db?host=/cloudsql/YOUR_PROJECT_ID:us-central1:electricity-db"
   ```

### Deployment Option 2: Compute Engine

1. **Create VM Instance**
   ```bash
   gcloud compute instances create electricity-server \
     --image-family ubuntu-2004-lts \
     --image-project ubuntu-os-cloud \
     --machine-type n1-standard-1 \
     --zone us-central1-a
   ```

2. **SSH into Instance**
   ```bash
   gcloud compute ssh electricity-server --zone us-central1-a
   ```

3. **Install Dependencies** (same as AWS EC2 steps)

4. **Deploy Application** (same as AWS EC2 steps)

### Deployment Option 3: GKE (Kubernetes)

See [Kubernetes](#kubernetes) section below for GKE-specific deployment.

### Setup Cloud Monitoring

```bash
gcloud monitoring dashboards create --config-from-file=monitoring-dashboard.json
```

### View Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=spain-electricity-analysis" --limit 50 --format json
```

---

## Kubernetes

### Prerequisites

- kubectl installed
- Kubernetes cluster (local, cloud, or managed service)
- Docker image built and pushed to registry
- Helm (optional but recommended)

### Deployment Option 1: Manual Kubernetes Manifests

1. **Create Namespace**
   ```bash
   kubectl create namespace electricity
   ```

2. **Create ConfigMap** (configmap.yaml)
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: electricity-config
     namespace: electricity
   data:
     FLASK_ENV: production
   ```

3. **Create Secret** (secret.yaml)
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: electricity-secret
     namespace: electricity
   type: Opaque
   stringData:
     DATABASE_URL: postgresql://user:password@db-host:5432/electricity_db
     SECRET_KEY: your-secret-key-here
   ```

4. **Create PostgreSQL StatefulSet** (postgres-statefulset.yaml)
   ```yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: postgres-pvc
     namespace: electricity
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi
   ---
   apiVersion: apps/v1
   kind: StatefulSet
   metadata:
     name: postgres
     namespace: electricity
   spec:
     serviceName: postgres
     replicas: 1
     selector:
       matchLabels:
         app: postgres
     template:
       metadata:
         labels:
           app: postgres
       spec:
         containers:
         - name: postgres
           image: postgres:14-alpine
           ports:
           - containerPort: 5432
           env:
           - name: POSTGRES_USER
             value: user
           - name: POSTGRES_PASSWORD
             valueFrom:
               secretKeyRef:
                 name: electricity-secret
                 key: DATABASE_PASSWORD
           - name: POSTGRES_DB
             value: electricity_db
           volumeMounts:
           - name: postgres-storage
             mountPath: /var/lib/postgresql/data
         volumes:
         - name: postgres-storage
           persistentVolumeClaim:
             claimName: postgres-pvc
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: postgres
     namespace: electricity
   spec:
     clusterIP: None
     ports:
     - port: 5432
       targetPort: 5432
     selector:
       app: postgres
   ```

5. **Create Application Deployment** (deployment.yaml)
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: electricity-app
     namespace: electricity
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: electricity-app
     template:
       metadata:
         labels:
           app: electricity-app
       spec:
         containers:
         - name: app
           image: YOUR_REGISTRY/spain-electricity-analysis:latest
           imagePullPolicy: IfNotPresent
           ports:
           - containerPort: 5000
           envFrom:
           - configMapRef:
               name: electricity-config
           - secretRef:
               name: electricity-secret
           resources:
             requests:
               memory: "256Mi"
               cpu: "250m"
             limits:
               memory: "512Mi"
               cpu: "500m"
           livenessProbe:
             httpGet:
               path: /health
               port: 5000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 5000
             initialDelaySeconds: 5
             periodSeconds: 5
   ```

6. **Create Service** (service.yaml)
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: electricity-service
     namespace: electricity
   spec:
     type: LoadBalancer
     ports:
     - port: 80
       targetPort: 5000
       protocol: TCP
     selector:
       app: electricity-app
   ```

7. **Create Ingress** (ingress.yaml)
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: electricity-ingress
     namespace: electricity
     annotations:
       cert-manager.io/cluster-issuer: letsencrypt-prod
   spec:
     ingressClassName: nginx
     tls:
     - hosts:
       - electricity.yourdomain.com
       secretName: electricity-tls
     rules:
     - host: electricity.yourdomain.com
       http:
         paths:
         - path: /
           pathType: Prefix
           backend:
             service:
               name: electricity-service
               port:
                 number: 80
   ```

8. **Deploy to Cluster**
   ```bash
   kubectl apply -f configmap.yaml
   kubectl apply -f secret.yaml
   kubectl apply -f postgres-statefulset.yaml
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f ingress.yaml
   ```

9. **Verify Deployment**
   ```bash
   kubectl get deployments -n electricity
   kubectl get pods -n electricity
   kubectl get svc -n electricity
   ```

### Deployment Option 2: Using Helm (Recommended)

1. **Create Helm Chart Structure**
   ```
   electricity-chart/
   ├── Chart.yaml
   ├── values.yaml
   ├── templates/
   │   ├── deployment.yaml
   │   ├── service.yaml
   │   ├── ingress.yaml
   │   ├── configmap.yaml
   │   └── secret.yaml
   ```

2. **Chart.yaml**
   ```yaml
   apiVersion: v2
   name: spain-electricity-analysis
   description: A Helm chart for Spain Electricity Analysis
   type: application
   version: 1.0.0
   appVersion: "1.0"
   ```

3. **values.yaml**
   ```yaml
   replicaCount: 3

   image:
     repository: YOUR_REGISTRY/spain-electricity-analysis
     tag: latest
     pullPolicy: IfNotPresent

   service:
     type: LoadBalancer
     port: 80
     targetPort: 5000

   ingress:
     enabled: true
     hostname: electricity.yourdomain.com
     tls:
       enabled: true
       issuer: letsencrypt-prod

   resources:
     requests:
       memory: "256Mi"
       cpu: "250m"
     limits:
       memory: "512Mi"
       cpu: "500m"

   postgresql:
     enabled: true
     auth:
       username: user
       password: changeme
       database: electricity_db

   env:
     FLASK_ENV: production
   ```

4. **Install Helm Chart**
   ```bash
   helm repo add myrepo YOUR_HELM_REPO_URL
   helm install electricity myrepo/spain-electricity-analysis \
     --namespace electricity \
     --create-namespace \
     --values values.yaml
   ```

5. **Upgrade Helm Chart**
   ```bash
   helm upgrade electricity myrepo/spain-electricity-analysis \
     --namespace electricity \
     --values values.yaml
   ```

### Kubernetes Management Commands

```bash
# View pod logs
kubectl logs -n electricity deployment/electricity-app

# Port forward for local testing
kubectl port-forward -n electricity svc/electricity-service 8080:80

# Scale deployment
kubectl scale deployment electricity-app -n electricity --replicas=5

# Rolling update
kubectl set image deployment/electricity-app \
  app=YOUR_REGISTRY/spain-electricity-analysis:v2 \
  -n electricity

# Check resource usage
kubectl top nodes
kubectl top pods -n electricity

# Delete deployment
kubectl delete namespace electricity
```

### GKE-Specific Setup

1. **Create GKE Cluster**
   ```bash
   gcloud container clusters create electricity-cluster \
     --zone us-central1-a \
     --num-nodes 3 \
     --enable-autoscaling \
     --min-nodes 2 \
     --max-nodes 10
   ```

2. **Get Cluster Credentials**
   ```bash
   gcloud container clusters get-credentials electricity-cluster \
     --zone us-central1-a
   ```

3. **Configure kubectl**
   ```bash
   gcloud container clusters get-credentials electricity-cluster
   ```

---

## Best Practices

### Security
- Use environment variables for sensitive data
- Implement HTTPS/TLS for all deployments
- Use private container registries
- Implement proper RBAC policies in Kubernetes
- Regularly update dependencies

### Monitoring & Logging
- Setup centralized logging (ELK, Stackdriver, CloudWatch)
- Monitor CPU, memory, and disk usage
- Setup alerts for critical metrics
- Track application performance metrics

### Scaling & Load Balancing
- Use load balancers for distributing traffic
- Implement horizontal pod autoscaling in Kubernetes
- Cache frequently accessed data
- Use CDN for static assets

### Database Management
- Regular automated backups
- Use connection pooling
- Implement read replicas for scaling reads
- Monitor query performance

### CI/CD Integration
- Automate testing before deployment
- Use blue-green deployments
- Implement canary releases
- Maintain version control for all configurations

---

## Troubleshooting

### Common Issues

**Application won't start**
- Check environment variables are set correctly
- Verify database connectivity
- Review application logs
- Ensure all dependencies are installed

**Database connection errors**
- Verify DATABASE_URL format
- Check database server is running and accessible
- Verify database credentials
- Ensure network security groups/firewall rules allow traffic

**Performance issues**
- Check resource allocation
- Monitor database query performance
- Implement caching strategies
- Scale horizontally if needed

**Port/binding issues**
- Ensure port isn't already in use
- Check firewall rules
- Verify port mapping in container/k8s config
- Check security group rules (AWS) or network policies (GCP)

---

## Support & Documentation

For additional help:
- GitHub Issues: [github.com/xczernia/spain-electricity-analysis/issues](https://github.com/xczernia/spain-electricity-analysis/issues)
- Documentation: Check README.md
- Contributing: See CONTRIBUTING.md

---

**Last Updated**: January 5, 2026
**Maintainer**: xczernia
