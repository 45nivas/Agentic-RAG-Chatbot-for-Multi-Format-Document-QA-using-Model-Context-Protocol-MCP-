#!/bin/bash
# Deployment script for RAG Chatbot

echo "🚀 Starting RAG Chatbot Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and configurations"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads chroma_db

# Build and start the application
echo "🏗️  Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check health
echo "🏥 Checking application health..."
if curl -f http://localhost:5000/ &> /dev/null; then
    echo "✅ RAG Chatbot is running successfully!"
    echo "🌐 Access your application at: http://localhost:5000"
    echo "📊 View logs with: docker-compose logs -f"
else
    echo "❌ Application is not responding. Check logs:"
    docker-compose logs
fi

echo "🎉 Deployment complete!"
