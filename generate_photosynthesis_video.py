import os
import model
from config import app_config
import mongo_utils as mongo

def generate_photosynthesis_video():
    # Image path
    image_file = "dataset/upload/image1.jpeg"
    
    # Prompt to ensure it's understandable for students
    prompt = (
        "Create an engaging educational story about photosynthesis for students. "
        "Explain the process of how plants make their own food using sunlight, water, and carbon dioxide, "
        "and how they release oxygen. Use a clear, friendly, and encouraging tone suitable for a middle-school student. "
        "The story should be visually descriptive to help the AI generate a great 3D animation."
    )
    
    genre = "3D Animation"
    style = "3D Animation"
    language = "English"
    word_count = 80
    creativity = 0.7
    engine = "Google Veo"
    
    print(f"Starting photosynthesis story and video generation for {image_file}...")
    
    try:
        # Generate the story and video
        story, video_path, source, max_c, curr_c, avail_c = model.generate_video(
            source_file=image_file,
            prompt=prompt,
            genre=genre,
            style=style,
            language=language,
            word_count=word_count,
            creativity=creativity,
            engine=engine
        )
        
        print("\n" + "="*50)
        print("GENERATED STORY:")
        print("="*50)
        print(story)
        print("="*50)
        
        if video_path:
            print(f"\nSUCCESS: Video generated!")
            print(f"Video Path/URL: {video_path}")
        else:
            print(f"\nFAILED: Video generation failed.")
            
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    # Initialize access count
    mongo.fetch_curr_access_count()
    generate_photosynthesis_video()
