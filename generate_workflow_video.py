import os
import fal_client
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

def generate_workflow_video():
    # Construct the detailed prompt from user requirements
    prompt = (
        "Clean animated explainer video showing a system workflow. "
        "Scene: modern UI interface with smooth transitions. "
        "Elements: AI Engine block in center; Image input, Generated Story, Cloud Storage aligned horizontally below. "
        "Arrows showing data flow (unidirectional, smooth animation). "
        "Minimal design, white background. "
        "Style: flat design, modern tech UI, professional. "
        "Animation: Elements appear one by one, Arrows animate with flow direction, Smooth transitions. "
        "Color theme: soft blue, white, minimal contrast. "
        "Quality: sharp, clean, presentation-ready, 4K resolution, highly detailed 3D visualization."
    )
    
    print("Starting generation of the system workflow video...")
    try:
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
            print(f"\nSUCCESS: Workflow video generated!")
            print(f"URL: {video_url}")
        else:
            print(f"\nFAILED: Could not find video URL in result: {result}")
            
    except Exception as e:
        print(f"\nERROR: Video generation failed: {str(e)}")

if __name__ == "__main__":
    generate_workflow_video()
