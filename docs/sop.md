# Standard Operating Procedure (SOP)

## 1. Deployment
- Use `docker-compose up --build` for local/dev
- For production, apply K8s manifests in `k8s/`:
  - Deploy ML pipeline, Raft simulator, chatbot, and MLflow
  - Apply HPA and monitoring configs

## 2. Configuration
- Set API keys and secrets via environment variables or K8s secrets
- Update service endpoints as needed

## 3. Monitoring
- Use Prometheus and Grafana for metrics and dashboards
- Review alert rules in `k8s/monitoring.yml`
- Respond to alerts for downtime, errors, or high latency

## 4. Scaling
- HPA will autoscale services based on CPU
- Monitor resource usage and adjust limits as needed

## 5. Security
- Rotate API keys regularly
- Enable RBAC and network policies in K8s
- Use HTTPS for all endpoints in production

## 6. Maintenance
- Regularly update dependencies and images
- Run end-to-end and load tests before major changes
- Backup MLflow and other persistent data

## 7. Troubleshooting
- Check logs for each service (`docker-compose logs` or `kubectl logs`)
- Restart services if unhealthy
- Consult documentation and team for unresolved issues 