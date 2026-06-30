import os
import fal_client
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("FAL_KEY")
print(f"Key loaded: {bool(key)}")

if key:
    image_path = "assets/examples/teacher-school.jpg"
    print(f"Uploading {image_path}...")
    try:
        url = fal_client.upload_file(image_path)
        print(f"Upload successful: {url}")
    except Exception as e:
        print(f"Upload failed: {e}")
