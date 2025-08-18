#!/bin/bash
set -xe  # Enable command tracing and exit on error

# Variables
AWS_REGION="ap-south-1"
AWS_ACCOUNT_ID="739275446561"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="prashant-mlops-ecr"
IMAGE_LOCAL_NAME="prashant-mlops-ecr"
IMAGE_TAG="latest"
IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"
CONTAINER_NAME="prashant-mlops-container"

echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

echo "Stopping and removing any existing container..."
docker rm -f ${CONTAINER_NAME} || true

echo "Pulling latest image from ECR..."
docker pull ${IMAGE_NAME}

echo "Running new container..."
docker run -d -p 80:5000 \
  -e DAGSHUB_PAT=7bed6b5be2021b1a4eaae221787bcb048ab2bcfd \
  --name ${CONTAINER_NAME} \
  ${IMAGE_NAME}

echo "Container started successfully. Logs:"
docker logs -f ${CONTAINER_NAME}