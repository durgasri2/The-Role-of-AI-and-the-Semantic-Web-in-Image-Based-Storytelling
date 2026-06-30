import os
import fal_client
import base64
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("FAL_KEY")
print(f"Key loaded: {bool(key)}")

if key:
    image_path = "assets/examples/teacher-school.jpg"
    with open(image_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{base64_image}"
    
    print(f"Subscribing to fal-ai/luma-dream-machine/ray-2/image-to-video with data URI...")
    try:
        # We'll just try a very short generation or just a test subscribe if possible
        # Actually, let's try a simple one
        result = fal_client.subscribe(
            "fal-ai/luma-dream-machine/ray-2/image-to-video",
            arguments={
                "image_url": image_url,
                "prompt": "Lumiere style lifelike motion, cinematic 3D animation",
            },
            with_logs=True,
        )
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
