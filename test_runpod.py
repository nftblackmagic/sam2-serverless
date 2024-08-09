import requests
import json
import time
import sys

# Replace these with your actual values
ENDPOINT_ID = "9o7a16v9j2tiqm"
API_KEY = ""

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

def send_request(input_data):
    url = f"{BASE_URL}/run"
    response = requests.post(url, headers=HEADERS, json=input_data)
    return response.json()

def check_status(job_id):
    url = f"{BASE_URL}/status/{job_id}"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def main(input_file):
    # Read input from JSON file
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    print("Input data:", json.dumps(input_data, indent=2))
    print("\nSending initial request...")
    response = send_request(input_data)
    print("Initial response:", json.dumps(response, indent=2))

    if 'id' not in response:
        print("Failed to get a job ID. Exiting.")
        return

    job_id = response['id']
    print(f"\nJob ID: {job_id}")

    while True:
        time.sleep(5)  # Wait for 5 seconds before checking status
        status_response = check_status(job_id)
        print("\nStatus response:", json.dumps(status_response, indent=2))

        if 'status' in status_response:
            if status_response['status'] == 'COMPLETED':
                print("\nJob completed successfully!")
                if 'output' in status_response:
                    print("\nOutput:")
                    print(json.dumps(status_response['output'], indent=2))
                break
            elif status_response['status'] in ['FAILED', 'CANCELLED']:
                print(f"\nJob {status_response['status'].lower()}.")
                if 'error' in status_response:
                    print("\nError:")
                    print(json.dumps(status_response['error'], indent=2))
                break
        else:
            print("Unexpected response format. Continuing to poll...")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)