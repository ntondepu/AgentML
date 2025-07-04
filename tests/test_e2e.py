import requests

API_KEY = "demo-key-123"
HEADERS = {"X-API-Key": API_KEY}

# ML Pipeline
def test_ml_pipeline():
    r = requests.post("http://localhost:8001/run-pipeline", headers=HEADERS)
    assert r.status_code == 200
    print("ML pipeline run triggered.")
    r = requests.get("http://localhost:8001/latest-accuracy", headers=HEADERS)
    assert r.status_code == 200
    print("ML pipeline accuracy:", r.json())

# Raft Simulator
def test_raft():
    r = requests.get("http://localhost:8002/api/raft/cluster-state", headers=HEADERS)
    assert r.status_code == 200
    print("Raft cluster state:", r.json())
    r = requests.post("http://localhost:8002/api/raft/trigger-election", headers=HEADERS)
    assert r.status_code == 200
    print("Raft election triggered.")

# Chatbot
def test_chatbot():
    r = requests.post("http://localhost:8003/chat", json={"user": "test", "message": "Hello!"}, headers=HEADERS)
    assert r.status_code == 200
    print("Chatbot response:", r.json())

if __name__ == "__main__":
    test_ml_pipeline()
    test_raft()
    test_chatbot() 