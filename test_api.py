import requests

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    def print_response(endpoint, response):
        print(f"\n=== Testing {endpoint} ===")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
        print("=" * 50)

    # Test root endpoint
    response = requests.get(f"{base_url}/")
    print_response("Root Endpoint", response)
    
    # Test products endpoint
    response = requests.get(f"{base_url}/products")
    print_response("Products Endpoint", response)
    
    # Test single product endpoint
    response = requests.get(f"{base_url}/products/P1")
    print_response("Single Product Endpoint", response)
    
    # Test inventory status endpoint
    response = requests.get(f"{base_url}/inventory/status")
    print_response("Inventory Status Endpoint", response)
    
    # Test supply chain metrics endpoint
    response = requests.get(f"{base_url}/supply-chain/metrics")
    print_response("Supply Chain Metrics Endpoint", response)

if __name__ == "__main__":
    test_api()