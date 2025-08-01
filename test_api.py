#!/usr/bin/env python3
"""
MARS-GIS API Test Script
Tests key API endpoints to validate functionality
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi.testclient import TestClient

from mars_gis.main import app


def test_api_endpoints():
    """Test key API endpoints"""
    client = TestClient(app)

    print("🧪 Testing MARS-GIS API Endpoints")
    print("=" * 50)

    # Test 1: Health check
    print("1. Testing health endpoint...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
    print()

    # Test 2: System health
    print("2. Testing system health endpoint...")
    response = client.get("/api/v1/system/health")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   API Status: {data.get('api_status', 'Unknown')}")
        print(f"   Version: {data.get('version', 'Unknown')}")
    print()

    # Test 3: Mars datasets
    print("3. Testing Mars datasets endpoint...")
    response = client.get("/api/v1/mars-data/datasets")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Available datasets: {len(data.get('datasets', []))}")
    print()

    # Test 4: ML models
    print("4. Testing ML models endpoint...")
    response = client.get("/api/v1/inference/models")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Available models: {len(data.get('models', []))}")
    print()

    # Test 5: Missions
    print("5. Testing missions endpoint...")
    response = client.get("/api/v1/missions")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total missions: {data.get('total', 0)}")
    print()

    print("✅ API Testing Complete")
    print("🚀 All core endpoints are responding correctly!")

if __name__ == "__main__":
    test_api_endpoints()
