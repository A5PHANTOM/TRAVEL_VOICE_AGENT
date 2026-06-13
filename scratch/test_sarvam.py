import urllib.request
import json
import time

def test_sarvam():
    api_key = "sk_uo5w0xqm_vtuJYCyHAkQVzHODGkpCuBbz"
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "text": "Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?",
        "target_language_code": "en-IN",
        "speaker": "priya",
        "sample_rate": 24000,
        "enable_preprocessing": True,
        "model": "bulbul:v3",
        "pace": 1.0,
    }
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    print("Sending request to Sarvam TTS via urllib...")
    start_time = time.time()
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers, 
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.status
            print(f"Status: {status}")
            if status == 200:
                data = json.loads(response.read().decode('utf-8'))
                print(f"Success! Received {len(data.get('audios', []))} audios.")
                print(f"Time taken: {time.time() - start_time:.2f} seconds")
            else:
                print(f"Failed: {response.read()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sarvam()
