"""
test_service_integration.py

Exercises the containerized /chat endpoint end-to-end. Run this against a
live container (docker run already started) to check the paths that
haven't been tested yet through the actual HTTP/container layer.

Usage:
  python3 test_service_integration.py [base_url] [test_image_path]

  base_url defaults to http://localhost:9000
  test_image_path is optional — image test is skipped if not provided
"""

import sys
import json
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9000"
TEST_IMAGE = sys.argv[2] if len(sys.argv) > 2 else None


def show(label, resp):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"Status: {resp.status_code}")
    try:
        body = resp.json()
        print(json.dumps(body, indent=2)[:1000])
        return body
    except Exception:
        print(resp.text[:500])
        return None


def test_health():
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    body = show("1. Health check", resp)
    assert resp.status_code == 200 and body.get("engine_loaded") is True, "Engine not loaded"
    print("PASS")


def test_text_query():
    resp = requests.post(f"{BASE_URL}/chat", json={"text": "What do BSF larvae eat?"}, timeout=120)
    body = show("2. Plain text query", resp)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(body["response"]) > 20, "Response suspiciously short"
    assert isinstance(body["history"], list) and len(body["history"]) == 2, "History should have 2 turns after 1 exchange"
    print("PASS")
    return body["history"]


def test_history_round_trip(history):
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"text": "How long does that stage take?", "history": history},
        timeout=120,
    )
    body = show("3. Multi-turn history round-trip", resp)
    assert resp.status_code == 200
    assert len(body["history"]) == 4, f"Expected 4 turns (2 exchanges), got {len(body['history'])}"
    print("PASS - history correctly accumulated across calls")


def test_domain_gate():
    resp = requests.post(f"{BASE_URL}/chat", json={"text": "What's the capital of France?"}, timeout=30)
    body = show("4. Domain-scope gate (off-topic query)", resp)
    assert resp.status_code == 200
    refused = "outside what i can help" in body["response"].lower() or "bsf" in body["response"].lower()
    if refused:
        print("PASS - off-topic query correctly refused")
    else:
        print("WARNING: off-topic query may have slipped through the gate - check response above")


def test_reset():
    resp = requests.post(f"{BASE_URL}/chat", json={"text": "clear"}, timeout=30)
    body = show("5. Reset command", resp)
    assert resp.status_code == 200
    assert body["history"] == [], "History should be empty after reset"
    print("PASS")


def test_image_query():
    if not TEST_IMAGE:
        print("\n" + "="*60 + "\n6. Image query - SKIPPED (no test image path provided)\n" + "="*60)
        return
    import base64
    with open(TEST_IMAGE, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"text": "Identify the life stage and advise accordingly.", "image_base64": image_b64},
        timeout=180,
    )
    body = show("6. Image query", resp)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(body["response"]) > 20
    print("PASS")


def test_unreachable_llama_server_note():
    print("\n" + "="*60)
    print("7. llama-server-unreachable scenario - NOT automated")
    print("="*60)
    print("Manually: stop llama-server, then re-run test_text_query() alone.")
    print("Expected: a clean error message in the response, NOT a 500 crash.")
    print("This confirms the try/except around the HTTP call in generate_response()")
    print("actually degrades gracefully instead of propagating an exception.")


if __name__ == "__main__":
    print(f"Testing service at {BASE_URL}")
    test_health()
    history = test_text_query()
    test_history_round_trip(history)
    test_domain_gate()
    test_reset()
    test_image_query()
    test_unreachable_llama_server_note()
    print("\n" + "="*60)
    print("Automated tests complete. Review any WARNING lines above,")
    print("and run the manual llama-server-unreachable check separately.")
    print("="*60)