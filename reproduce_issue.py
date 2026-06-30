import model
import os
from dotenv import load_dotenv

load_dotenv()

source_file = "assets/examples/teacher-school.jpg"
prompt = ""
genre = "3D Animation"
style = "3D Animation"
language = "English"
word_count = 50
creativity = 0.7

print("Starting generate_video...")
try:
    result = model.generate_video(source_file, prompt, genre, style, language, word_count, creativity)
    print("Result:")
    print(f"Story: {result[0][:100]}...")
    print(f"Video Path: {result[1]}")
    if result[1] is None:
        print(f"Error Message: {result[0]}")
except Exception as e:
    print(f"Caught Exception: {e}")
