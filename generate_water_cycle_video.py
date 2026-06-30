import os
import fal_client
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

def generate_water_cycle_video():
    # Prompt provided by the user
    prompt = (
        "Create a detailed animated educational video of the water cycle that shows "
        "both the external scene and the internal microscopic processes happening inside it. "
        "Scene starts with a colorful cartoon landscape: sun, lake, mountains, clouds, and sky. "
        "Then smoothly zoom into the internal process of evaporation: show water molecules "
        "inside the lake gaining heat energy from the sun, moving faster, and escaping "
        "into the air as water vapor. Transition to condensation (internal view): zoom "
        "into the sky where water vapor molecules slow down, lose heat, and cluster "
        "together to form tiny droplets, building clouds. Next, show precipitation "
        "(internal process): inside the cloud, tiny droplets collide and combine into "
        "larger, heavier drops until gravity pulls them down as rain. Then show collection "
        "and flow (internal view): rainwater soaking into soil (infiltration), flowing "
        "through underground layers, rivers, and returning to the lake. Use smooth "
        "zoom-in and zoom-out transitions between macro (landscape) and micro (molecular) "
        "views. Include particle-level animation of molecules, energy transfer, and motion. "
        "Keep it visually clear, educational, and engaging with labeled stages and arrows. "
        "Use soft colors and realistic physics-based motion."
    )
    
    print("Starting generation of the water cycle video...")
    try:
        # Using fal-ai/veo3.1 as it was used in generate_workflow_video.py
        result = fal_client.subscribe(
            "fal-ai/veo3.1",
            arguments={
                "prompt": prompt,
                "resolution": "1080p",
                "aspect_ratio": "16:9",
                "generate_audio": True
            },
            with_logs=True,
        )
        
        video_url = None
        if 'video' in result and 'url' in result['video']:
            video_url = result['video']['url']
        elif 'url' in result:
            video_url = result['url']
            
        if video_url:
            print(f"\nSUCCESS: Water cycle video generated!")
            print(f"URL: {video_url}")
            # Optionally, you could download it here if needed
        else:
            print(f"\nFAILED: Could not find video URL in result: {result}")
            
    except Exception as e:
        print(f"\nERROR: Video generation failed: {str(e)}")

if __name__ == "__main__":
    generate_water_cycle_video()
