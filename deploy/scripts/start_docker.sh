#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 739275446561.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 739275446561.dkr.ecr.ap-south-1.amazonaws.com/test-mlops/yt-chrome-plugin:latest

echo "Removing all existing containers..."
docker rm -f $(docker ps -aq) 2>/dev/null || true

echo "Starting new container..."
docker run -d -p 80:5000 --name prashant-mlops-container 739275446561.dkr.ecr.ap-south-1.amazonaws.com/test-mlops/yt-chrome-plugin:latest

echo "Container started successfully."