#!/bin/bash
#
# Docker Setup Script for Open-Instruct
# This script builds the Docker image for training
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Image name
IMAGE_NAME="${IMAGE_NAME:-open-instruct}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Get git info for build args
GIT_COMMIT=$(git -C /home/mohamed/rl_curriculum/open-instruct rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git -C /home/mohamed/rl_curriculum/open-instruct rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# ============================================================================
# Build Docker Image
# ============================================================================

echo "============================================"
echo "Building Docker Image for Open-Instruct"
echo "============================================"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Git Commit: ${GIT_COMMIT}"
echo "Git Branch: ${GIT_BRANCH}"
echo "============================================"
echo ""

cd /home/mohamed/rl_curriculum/open-instruct

docker build . \
    --build-arg GIT_COMMIT=${GIT_COMMIT} \
    --build-arg GIT_BRANCH=${GIT_BRANCH} \
    -t ${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "============================================"
echo "Docker image built successfully!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "============================================"

