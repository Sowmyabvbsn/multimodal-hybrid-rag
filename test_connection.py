#!/usr/bin/env python3
"""
Test script to debug Qdrant connection issues
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import requests
import socket

def test_basic_connectivity():
    """Test basic network connectivity"""
    print("🔍 Testing basic connectivity...")
    
    # Load environment variables
    load_dotenv()
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"QDRANT_URL: {qdrant_url}")
    print(f"QDRANT_API_KEY: {'*' * 20 if qdrant_api_key else 'Not set'}")
    
    if not qdrant_url:
        print("❌ QDRANT_URL not found in environment variables")
        return False
    
    # Parse URL to get host and port
    try:
        if qdrant_url.startswith('https://'):
            host = qdrant_url.replace('https://', '').split(':')[0]
            port = int(qdrant_url.split(':')[-1]) if ':' in qdrant_url.split('//')[1] else 443
        elif qdrant_url.startswith('http://'):
            host = qdrant_url.replace('http://', '').split(':')[0]
            port = int(qdrant_url.split(':')[-1]) if ':' in qdrant_url.split('//')[1] else 80
        else:
            print("❌ Invalid URL format")
            return False
            
        print(f"Host: {host}")
        print(f"Port: {port}")
        
        # Test DNS resolution
        print("\n🔍 Testing DNS resolution...")
        try:
            ip = socket.gethostbyname(host)
            print(f"✅ DNS resolved: {host} -> {ip}")
        except socket.gaierror as e:
            print(f"❌ DNS resolution failed: {e}")
            return False
        
        # Test port connectivity
        print(f"\n🔍 Testing port connectivity to {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        try:
            result = sock.connect_ex((host, port))
            if result == 0:
                print(f"✅ Port {port} is open")
            else:
                print(f"❌ Port {port} is closed or filtered")
                return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
        finally:
            sock.close()
        
        return True
        
    except Exception as e:
        print(f"❌ URL parsing failed: {e}")
        return False

def test_http_request():
    """Test HTTP request to Qdrant"""
    print("\n🔍 Testing HTTP request...")
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    try:
        # Test basic HTTP request
        headers = {}
        if qdrant_api_key:
            headers['api-key'] = qdrant_api_key
        
        response = requests.get(f"{qdrant_url}/collections", headers=headers, timeout=10)
        print(f"HTTP Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ HTTP request successful")
            return True
        else:
            print(f"❌ HTTP request failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP request failed: {e}")
        return False

def test_qdrant_client():
    """Test Qdrant client connection"""
    print("\n🔍 Testing Qdrant client...")
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections = client.get_collections()
        print(f"✅ Qdrant client connected successfully")
        print(f"Found {len(collections.collections)} collections")
        for collection in collections.collections:
            print(f"  - {collection.name}")
        return True
        
    except Exception as e:
        print(f"❌ Qdrant client connection failed: {e}")
        return False

def main():
    """Run all connection tests"""
    print("🚀 Qdrant Connection Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    if not test_basic_connectivity():
        print("\n❌ Basic connectivity failed. Check your network and URL.")
        sys.exit(1)
    
    # Test 2: HTTP request
    if not test_http_request():
        print("\n❌ HTTP request failed. Check your API key and URL.")
        sys.exit(1)
    
    # Test 3: Qdrant client
    if not test_qdrant_client():
        print("\n❌ Qdrant client failed. Check your configuration.")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Your Qdrant connection is working.")

if __name__ == "__main__":
    main()