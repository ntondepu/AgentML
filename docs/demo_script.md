# Demo Walkthrough Script

## 1. Start All Services
- Run: `docker-compose up --build`
- Wait for all containers to be healthy (ml-pipeline, raft-simulator, chatbot, mlflow)

## 2. Open the Frontend
- Navigate to the React app in your browser (or run `npm start` in `frontend/`)
- Show the dashboard with ML pipeline, distributed sim, and chatbot tabs

## 3. ML Pipeline Demo
- Go to the ML Pipeline tab
- Trigger a new pipeline run
- Show progress and metrics (accuracy)
- Open MLflow UI at http://localhost:5000 to show experiment tracking

## 4. Raft Simulator Demo
- Go to the Distributed Sim tab
- Visualize the Raft cluster (leader, followers, partitions)
- Trigger an election and create a partition
- Show real-time updates in the UI

## 5. AI Chatbot Demo
- Go to the AI Chatbot tab
- Type a question (e.g., "What's the latest model accuracy?")
- Show the assistant's response
- Ask about cluster status or trigger actions via chat

## 6. Monitoring & Alerts
- Open Grafana/Prometheus dashboards (if set up)
- Show service health, error rates, and alerts

## 7. Wrap Up
- Highlight modularity, extensibility, and observability
- Invite questions and feedback 