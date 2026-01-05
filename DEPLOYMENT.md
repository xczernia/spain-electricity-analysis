# Deployment Guide

This guide provides comprehensive instructions for deploying the Spain Electricity Analysis project across multiple platforms.

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Heroku Deployment](#heroku-deployment)
4. [AWS Deployment](#aws-deployment)
5. [Google Cloud Deployment](#google-cloud-deployment)
6. [Environment Variables](#environment-variables)
7. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Virtual environment tool (venv or conda)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/xczernia/spain-electricity-analysis.git
   cd spain-electricity-analysis
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   
   # Or using conda
   conda create -n spain-electricity python=3.8+
   conda activate spain-electricity
   ```

3. **Activate the virtual environment**
   ```bash
   # On macOS/Linux
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   nano .env
   ```

6. **Initialize the database** (if applicable)
   ```bash
   python manage.py db upgrade
   # or
   flask db upgrade
   ```

7. **Run the application**
   ```bash
   python app.py
   # or
   flask run
   ```

8. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

### Development Commands

```bash
# Run tests
pytest

# Run with debug mode
FLASK_ENV=development flask run

# Run specific tests
pytest tests/test_electricity.py

# Generate coverage report
pytest --cov=app tests/
```

---

## Docker Deployment

### Prerequisites

- Docker (version 20.10+)
- Docker Compose (version 1.29+)

### Quick Start with Docker Compose

1. **Clone the repository**
   ```bash
   git clone https://github.com/xczernia/spain-electricity-analysis.git
   cd spain-electricity-analysis
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   ```
   http://localhost:5000
   ```

5. **View logs**
   ```bash
   docker-compose logs -f app
   ```

6. **Stop the containers**
   ```bash
   docker-compose down
   ```

### Manual Docker Build

1. **Build the Docker image**
   ```bash
   docker build -t spain-electricity:latest .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name spain-electricity \
     -p 5000:5000 \
     --env-file .env \
     spain-electricity:latest
   ```

3. **View running containers**
   ```bash
   docker ps
   ```

4. **Stop the container**
   ```bash
   docker stop spain-electricity
   docker rm spain-electricity
   ```

### Docker Compose with Services

```yaml
# docker-compose.yml structure
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    depends_on:
      - db
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=spain_electricity
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Heroku Deployment

### Prerequisites

- Heroku account (free tier available)
- Heroku CLI installed
- Git repository initialized

### Step-by-Step Deployment

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create a Heroku app**
   ```bash
   heroku create spain-electricity-analysis
   ```

4. **Add Procfile** (if not already present)
   ```bash
   echo "web: gunicorn app:app" > Procfile
   ```

5. **Add runtime.txt** (specify Python version)
   ```bash
   echo "python-3.9.13" > runtime.txt
   ```

6. **Set environment variables**
   ```bash
   heroku config:set FLASK_ENV=production
   heroku config:set DATABASE_URL=<your-database-url>
   heroku config:set SECRET_KEY=<your-secret-key>
   heroku config:set API_KEY=<your-api-key>
   ```

7. **Deploy to Heroku**
   ```bash
   git push heroku main
   ```

8. **Monitor deployment**
   ```bash
   heroku logs --tail
   ```

9. **Access your app**
   ```bash
   heroku open
   ```

### Heroku Add-ons

Optional add-ons for enhanced functionality:

```bash
# PostgreSQL database
heroku addons:create heroku-postgresql:hobby-dev

# Redis for caching
heroku addons:create heroku-redis:premium-0

# Scheduler for background jobs
heroku addons:create scheduler:standard
```

### Updating Your App

```bash
# Push changes
git push heroku main

# Run migrations
heroku run python manage.py db upgrade

# Restart app
heroku restart
```

---

## AWS Deployment

### Prerequisites

- AWS account with active credentials
- AWS CLI configured
- EC2 key pair created
- Security groups configured

### Option 1: EC2 with Elastic Beanstalk

1. **Install Elastic Beanstalk CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize Elastic Beanstalk**
   ```bash
   eb init -p python-3.9 spain-electricity-analysis
   ```

3. **Create environment**
   ```bash
   eb create production-env
   ```

4. **Set environment variables**
   ```bash
   eb setenv FLASK_ENV=production DATABASE_URL=<url> SECRET_KEY=<key>
   ```

5. **Deploy**
   ```bash
   eb deploy
   ```

6. **Monitor**
   ```bash
   eb open
   eb logs
   eb status
   ```

### Option 2: EC2 Manual Setup

1. **Launch EC2 instance**
   - AMI: Ubuntu 20.04 LTS
   - Instance type: t3.micro (free tier)
   - Security group: Allow HTTP (80), HTTPS (443), SSH (22)

2. **Connect to instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Update and install dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip python3-venv nginx git
   ```

4. **Clone repository**
   ```bash
   git clone https://github.com/xczernia/spain-electricity-analysis.git
   cd spain-electricity-analysis
   ```

5. **Setup virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install gunicorn
   ```

6. **Configure Nginx**
   ```bash
   sudo nano /etc/nginx/sites-available/spain-electricity
   ```

   Sample Nginx configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

7. **Enable Nginx site**
   ```bash
   sudo ln -s /etc/nginx/sites-available/spain-electricity /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

8. **Create systemd service**
   ```bash
   sudo nano /etc/systemd/system/spain-electricity.service
   ```

   Content:
   ```ini
   [Unit]
   Description=Spain Electricity Analysis
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/spain-electricity-analysis
   Environment="PATH=/home/ubuntu/spain-electricity-analysis/venv/bin"
   ExecStart=/home/ubuntu/spain-electricity-analysis/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5000 app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

9. **Start service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start spain-electricity
   sudo systemctl enable spain-electricity
   ```

### Option 3: ECS (Docker Containers)

1. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name spain-electricity
   ```

2. **Build and push Docker image**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   
   docker build -t spain-electricity:latest .
   docker tag spain-electricity:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/spain-electricity:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/spain-electricity:latest
   ```

3. **Create ECS cluster and service** (via AWS Console or CloudFormation)

---

## Google Cloud Deployment

### Prerequisites

- Google Cloud account with active billing
- Google Cloud SDK installed
- `gcloud` CLI configured

### Option 1: Google App Engine

1. **Install Google Cloud SDK**
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Ubuntu/Debian
   curl https://sdk.cloud.google.com | bash
   
   # Windows
   # Download from https://cloud.google.com/sdk/docs/install
   ```

2. **Initialize gcloud**
   ```bash
   gcloud init
   gcloud auth application-default login
   ```

3. **Create app.yaml**
   ```yaml
   runtime: python39
   
   env: standard
   
   entrypoint: gunicorn -b :$PORT app:app
   
   env_variables:
     FLASK_ENV: "production"
   
   handlers:
   - url: /.*
     script: auto
   ```

4. **Deploy to App Engine**
   ```bash
   gcloud app deploy
   ```

5. **View logs**
   ```bash
   gcloud app logs read -n 50
   ```

6. **Access your app**
   ```bash
   gcloud app browse
   ```

### Option 2: Google Cloud Run

1. **Build Docker image**
   ```bash
   gcloud builds submit --tag gcr.io/<project-id>/spain-electricity
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy spain-electricity \
     --image gcr.io/<project-id>/spain-electricity \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars FLASK_ENV=production
   ```

3. **Set environment variables**
   ```bash
   gcloud run services update spain-electricity \
     --update-env-vars DATABASE_URL=<url>,SECRET_KEY=<key>
   ```

### Option 3: Google Kubernetes Engine (GKE)

1. **Create GKE cluster**
   ```bash
   gcloud container clusters create spain-electricity-cluster \
     --zone us-central1-a \
     --num-nodes 2
   ```

2. **Build and push image to Artifact Registry**
   ```bash
   gcloud builds submit --tag us-central1-docker.pkg.dev/<project>/spain-electricity/app
   ```

3. **Create Kubernetes deployment manifest** (deployment.yaml)
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: spain-electricity
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: spain-electricity
     template:
       metadata:
         labels:
           app: spain-electricity
       spec:
         containers:
         - name: app
           image: us-central1-docker.pkg.dev/<project>/spain-electricity/app
           ports:
           - containerPort: 5000
           env:
           - name: FLASK_ENV
             value: "production"
   ```

4. **Get cluster credentials**
   ```bash
   gcloud container clusters get-credentials spain-electricity-cluster --zone us-central1-a
   ```

5. **Deploy to GKE**
   ```bash
   kubectl apply -f deployment.yaml
   ```

6. **Expose service**
   ```bash
   kubectl expose deployment spain-electricity --type=LoadBalancer --port=80 --target-port=5000
   ```

---

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/spain_electricity
DB_HOST=localhost
DB_PORT=5432
DB_NAME=spain_electricity
DB_USER=admin
DB_PASSWORD=your-password

# API Configuration
API_KEY=your-api-key
API_SECRET=your-api-secret
API_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# External Services
SENDGRID_API_KEY=your-sendgrid-key
SLACK_WEBHOOK_URL=your-slack-webhook

# Feature Flags
ENABLE_CACHING=true
CACHE_TIMEOUT=3600
ENABLE_NOTIFICATIONS=false

# Security
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
SESSION_TIMEOUT=1800
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>
```

#### 2. Database Connection Issues
```bash
# Verify database credentials
psql -h localhost -U admin -d spain_electricity

# Check connection string
echo $DATABASE_URL
```

#### 3. Module Not Found Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Verify Python version
python --version
```

#### 4. Docker Build Failures
```bash
# Clean up Docker system
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t spain-electricity:latest .
```

#### 5. Heroku Deployment Issues
```bash
# Check Heroku logs
heroku logs --tail

# Restart dyno
heroku restart

# Check Procfile
cat Procfile
```

#### 6. AWS Elastic Beanstalk Issues
```bash
# SSH into instance
eb ssh

# View EB logs
eb logs

# Check application logs
cat /var/log/eb-activity.log
```

#### 7. Google Cloud Issues
```bash
# Check Cloud Run logs
gcloud run logs read spain-electricity

# Debug locally with Cloud Run emulator
functions-framework --target=app --debug
```

### Performance Optimization

- **Caching**: Enable Redis for better performance
- **Database indexing**: Ensure proper indexes on frequently queried columns
- **Load balancing**: Use load balancers (AWS ELB, Google Cloud Load Balancing)
- **CDN**: Consider CloudFront (AWS) or Cloud CDN (Google Cloud)
- **Monitoring**: Set up CloudWatch (AWS) or Cloud Monitoring (Google Cloud)

### Security Considerations

- Always use HTTPS in production
- Rotate API keys and secrets regularly
- Use environment variables for sensitive data
- Enable database encryption
- Set up firewall rules and security groups
- Regular security audits and dependency updates
- Enable logging and monitoring
- Use VPN for sensitive operations

---

## Support and Resources

- **Documentation**: See README.md for project overview
- **Issues**: Report bugs on GitHub Issues
- **Contributing**: See CONTRIBUTING.md for guidelines
- **License**: See LICENSE file

For deployment questions, please open an issue on GitHub or contact the maintainers.

---

**Last Updated**: 2026-01-05
**Maintained by**: xczernia
