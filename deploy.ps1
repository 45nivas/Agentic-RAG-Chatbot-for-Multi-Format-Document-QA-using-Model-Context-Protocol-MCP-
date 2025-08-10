# PowerShell Deployment Script for Windows
# RAG Chatbot Docker Deployment

Write-Host "🚀 Starting RAG Chatbot Deployment..." -ForegroundColor Green

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker Compose is not installed. Please install Docker Compose first." -ForegroundColor Red
    exit 1
}

# Create environment file if it doesn't exist
if (!(Test-Path .env)) {
    Write-Host "📝 Creating environment file..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  Please edit .env file with your API keys and configurations" -ForegroundColor Yellow
}

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path uploads, chroma_db | Out-Null

# Build and start the application
Write-Host "🏗️  Building Docker images..." -ForegroundColor Cyan
docker-compose build

Write-Host "🚀 Starting services..." -ForegroundColor Cyan
docker-compose up -d

# Wait for services to be ready
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check health
Write-Host "🏥 Checking application health..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ RAG Chatbot is running successfully!" -ForegroundColor Green
        Write-Host "🌐 Access your application at: http://localhost:5000" -ForegroundColor Green
        Write-Host "📊 View logs with: docker-compose logs -f" -ForegroundColor Cyan
    }
} catch {
    Write-Host "❌ Application is not responding. Check logs:" -ForegroundColor Red
    docker-compose logs
}

Write-Host "🎉 Deployment complete!" -ForegroundColor Green
