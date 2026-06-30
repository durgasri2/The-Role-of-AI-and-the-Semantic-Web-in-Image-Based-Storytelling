from PIL import Image
try:
    img = Image.open("assets/examples/teacher-school.jpg")
    print(f"Image format: {img.format}, size: {img.size}, mode: {img.mode}")
except Exception as e:
    print(f"Failed to open image: {e}")
