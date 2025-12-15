import requests
import json
import time

BASE_URL = "https://agentsmedchat.onrender.com"

def test_health():
    print(f"Testing Health Check at {BASE_URL}/health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("[PASS] Health Check Passed!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"[FAIL] Health Check Failed with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] Error connecting to health endpoint: {e}")

def test_chat():
    print(f"\nTesting Chat Endpoint at {BASE_URL}/chat...")
    payload = {
        "query": "What are the common side effects of lisinopril?",
        "session_id": "test-session-deploy-001"
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            print(f"[PASS] Chat Request Successful ({duration:.2f}s)!")
            data = response.json()
            print("\n--- Response ---")
            print(f"Agent: {data.get('agent_type')}")
            print(f"Answer: {data.get('answer')}")
            print(f"Docs Retrieved: {len(data.get('retrieved_documents', []))}")
            print(f"Search Results: {len(data.get('search_results', []))}")
            if data.get('thinking_time'):
                print(f"Thinking Time: {data.get('thinking_time')}s")
        else:
            print(f"[FAIL] Chat Request Failed with status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] Error connecting to chat endpoint: {e}")

if __name__ == "__main__":
    print(f"Targeting: {BASE_URL}")
    test_health()
    test_chat()
