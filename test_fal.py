import os
import fal_client
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("FAL_KEY")
print(f"Key loaded: {bool(key)}")
if key:
    print(f"Key starts with: {key[:5]}...")
