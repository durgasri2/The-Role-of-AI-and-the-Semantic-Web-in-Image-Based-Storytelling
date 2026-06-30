import base64
import requests
import fal_client
import os
import asyncio
import edge_tts
import numpy as np
import cv2
import re
from moviepy import ImageClip, AudioFileClip, VideoClip, ColorClip, CompositeVideoClip, concatenate_videoclips

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from config import app_config
import mongo_utils as mongo


def __image2text(image):
    """Generates a short description of the image using Groq Vision or Hugging Face model"""
    # Try Groq Vision first if API key is available
    if app_config.GROQ_API_KEY:
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image).decode("utf-8")
            
            # Prepare request for Groq Vision
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {app_config.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image in detail. Detect and list: characters, objects, location, and the overall mood. Also, extract and list any text visible in the image (Extracted Text - OCR). Finally, provide a comprehensive summary of the setting, atmosphere, and actions. Present the result clearly with labels: 'Characters:', 'Objects:', 'Location:', 'Mood:', 'Extracted Text (OCR):', and 'Context Summary:'."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
            
            print(f"Groq Vision failed (status {response.status_code}), falling back to Hugging Face...")
        except Exception as e:
            print(f"Exception in Groq Vision: {str(e)}, falling back to Hugging Face...")

    # Fallback to Hugging Face
    if not app_config.I2T_API_URL or not app_config.HF_TOKEN:
        return "ERROR: No valid API key (Groq or HF) found for image captioning."

    try:
        headers = {"Authorization": f"Bearer {app_config.HF_TOKEN}"}
        response = requests.post(app_config.I2T_API_URL, headers=headers, data=image)
        
        if response.status_code != 200:
            return f"ERROR: Hugging Face API returned status {response.status_code}: {response.text[:200]}"
            
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
            
        return f"ERROR: Unexpected response format from Hugging Face API: {result}"
    except Exception as e:
        return f"ERROR: Exception in __image2text (Hugging Face): {str(e)}"


def __text2story(image_desc, prompt, genre, style, language, word_count, creativity):
    """Generates a short story based on image description text prompt"""
    if not app_config.GROQ_API_KEY and not app_config.OPENAI_KEY:
        return "ERROR: Groq or OpenAI API key not found in environment."
    
    if image_desc.startswith("ERROR:"):
        return f"Cannot generate story because image captioning failed: {image_desc}"

    ## chat LLM model
    if app_config.GROQ_API_KEY:
        story_model = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=app_config.GROQ_API_KEY,
            temperature=creativity,
        )
    else:
        story_model = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=app_config.OPENAI_KEY,
            temperature=creativity,
        )
    
    context = f"Image description: {image_desc}"
    if prompt:
        context += f"\nUser additional instructions: {prompt}"

    ## chat message prompts
    sys_prompt = PromptTemplate(
        template="""You are an expert educator and story writer specializing in educational 3D animation scripts. 
        Write a maximum of {word_count} words long explanation or story in {genre} genre 
        in {style} writing style, in {language} language, based on the user provided story-context.
        Focus on making the concepts clear, engaging, and easy for a student to understand.
        Break down complex ideas into simple, relatable terms.
        If the style is '3D Animation', write it as a highly engaging narrator script for a clear, high-quality 3D animation explainer, 
        using vivid analogies and clear step-by-step reasoning.
        """,
        input_variables=["word_count", "genre", "style", "language"],
    )
    system_msg_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)
    human_prompt = PromptTemplate(
        template="story-context: {context}", input_variables=["context"]
    )
    human_msg_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_msg_prompt, human_msg_prompt]
    )
    ## LLM chain
    story_chain = LLMChain(llm=story_model, prompt=chat_prompt)
    response = story_chain.run(
        genre=genre, style=style, language=language, word_count=word_count, context=context
    )
    return response


def __clean_script(text):
    """Cleans the narration script by removing labels, titles, and timestamps like (0-60s)"""
    # Keep the Title but remove the 'Title:' prefix
    text = re.sub(r'^(Title|TITLE):\s*', 'Title: ', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove Narration: or Narration line
    text = re.sub(r'^\s*(Narration|NARRATION):?\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove timestamps and seconds indicators (e.g., 60s, 60 seconds, [0-60s], (60s))
    text = re.sub(r'^\s*[\[\(]?\d+[-–]?\d*\s*(s|sec|seconds?)[\]\)]?:?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'[\[\(]\d+[-–]?\d*\s*(s|sec|seconds?)[\]\)]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+[-–]?\d*\s*(s|sec|seconds?)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\d+[-–]\d+\)', '', text)    # in parentheses without 's'
    
    # Remove labels like Hook:, Introduction:, etc. but KEEP Conclusion
    labels = ["Hook", "Introduction", "3D Animated Demonstration", "Animated Demonstration", "Visuals", "Visual Elements", "Writing"]
    for label in labels:
        text = re.sub(rf'^\d*\.?\s*{label}:', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Ensure 'Conclusion:' is formatted nicely if it exists
    text = re.sub(r'^(Conclusion|Conclusion Summary):\s*', '\nConclusion: ', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove extra whitespace and leading/trailing empty lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    return text


def __create_denoising_video(image_path, script_text, output_path="output_video.mp4", language="English"):
    """Creates a video showing the denoising process from noise to the image with subtitles"""
    print(f"Creating denoising video for script: {script_text[:50]}...")
    
    # 1. Generate Audio
    voice_map = {
        "English": "en-US-AndrewNeural",
        "Indian English": "en-IN-NeerjaNeural",
        "Hindi": "hi-IN-MadhurNeural",
        "Telugu": "te-IN-ShrutiNeural"
    }
    voice = voice_map.get(language, "en-US-AndrewNeural")
    communicate = edge_tts.Communicate(script_text, voice)
    audio_path = "temp_audio.mp3"
    
    # Safely save audio whether or not an event loop is running
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import threading
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(communicate.save(audio_path))
                new_loop.close()
            thread = threading.Thread(target=run_async)
            thread.start()
            thread.join()
        else:
            loop.run_until_complete(communicate.save(audio_path))
    except Exception:
        asyncio.run(communicate.save(audio_path))
        
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration
    
    # 2. Prepare Subtitles
    # Split text into sentences/phrases for subtitles
    sub_text_chunks = re.split(r'(?<=[.!?])\s+', script_text)
    sub_text_chunks = [s.strip() for s in sub_text_chunks if s.strip()]
    
    # 3. Load and Prepare Image
    if image_path.startswith("http"):
        # Download image
        response = requests.get(image_path)
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_path)
        
    if img is None:
        raise Exception(f"Failed to load image from {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to match capture format (1808, 902)
    img = cv2.resize(img, (1808, 902))
    h, w, _ = img.shape
    
    # 4. Create Denoising Frames
    noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    
    # The denoising will take 5 seconds or 30% of the video, whichever is smaller
    denoise_duration = min(5.0, total_duration * 0.5)
    
    def render_subtitles(frame, text):
        if not text:
            return frame
        
        # Define text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 # Larger font size for clarity
        thickness = 2 # Increased thickness for visibility
        color = (255, 255, 255) # White text
        
        # Wrap text to fit width
        max_width = w - 60
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            text_size = cv2.getTextSize(" ".join(current_line), font, font_scale, thickness)[0]
            if text_size[0] > max_width:
                if len(current_line) > 1:
                    lines.append(" ".join(current_line[:-1]))
                    current_line = [word]
                else:
                    lines.append(" ".join(current_line))
                    current_line = []
        
        if current_line:
            lines.append(" ".join(current_line))
            
        # Draw background and text for each line
        line_height = 35 # Increased line height for larger font
        y_start = h - 60 - (len(lines) - 1) * line_height
        
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = y_start + i * line_height
            
            # Draw semi-transparent black background for better visibility
            bg_rect_start = (text_x - 10, text_y - text_size[1] - 10)
            bg_rect_end = (text_x + text_size[0] + 10, text_y + 10)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw text on top
            cv2.putText(frame, line, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            
        return frame

    def make_frame(t):
        # Internal editing: Change zoom target every 3 seconds
        edit_cycle = 3.0
        cycle_idx = int(t / edit_cycle)
        
        # Base zoom/slide effect (Ken Burns)
        zoom_factor = 1.05 + 0.1 * (t / total_duration)
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        
        # Internal Editing: Dynamic focus points
        if cycle_idx % 3 == 1: # Focus top-left
            dx = int((new_w - w) * 0.2)
            dy = int((new_h - h) * 0.2)
        elif cycle_idx % 3 == 2: # Focus bottom-right
            dx = int((new_w - w) * 0.8)
            dy = int((new_h - h) * 0.8)
        else: # Center slide
            dx = int((new_w - w) * 0.5 * (1 + 0.2 * np.sin(t)))
            dy = int((new_h - h) * 0.5)
        
        # Base frame (denoising or final image)
        if t < denoise_duration:
            alpha = t / denoise_duration
            alpha = alpha * alpha 
            current_img = cv2.addWeighted(img, alpha, noise, 1 - alpha, 0)
        else:
            current_img = img.copy()

        # Apply Zoom & Pan (Internal Editing Effect)
        resized = cv2.resize(current_img, (new_w, new_h))
        # Crop back to (w, h) based on the "editing" coordinates
        x1 = dx
        y1 = dy
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, new_w - w))
        y1 = max(0, min(y1, new_h - h))
        base_frame = resized[y1:y1+h, x1:x1+w]
            
        # Add Subtitles if not Hindi or Telugu
        if language not in ["Hindi", "Telugu"]:
            chunk_index = int((t / total_duration) * len(sub_text_chunks))
            chunk_index = min(chunk_index, len(sub_text_chunks) - 1)
            
            if chunk_index >= 0:
                current_subtitle = sub_text_chunks[chunk_index]
                base_frame = render_subtitles(base_frame, current_subtitle)
            
        return base_frame

    video = VideoClip(make_frame, duration=total_duration)
    video = video.with_audio(audio)
    
    # 5. Write Output
    video.write_videofile(output_path, fps=30, codec="libx264", logger=None)
    
    # Cleanup
    audio.close()
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    return output_path


def __is_lfs_pointer(file_path):
    """Checks if a file is a Git LFS pointer instead of a real image"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(100)
            return "version https://git-lfs.github.com/spec/v1" in content
    except:
        return False


def __text2image(prompt):
    """Generates an image from a text prompt using fal-ai/flux/schnell"""
    print(f"Generating image from prompt: {prompt}")
    try:
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": f"{prompt}, Professional 3D Pixar-style story animation, highly detailed 3D characters and environment, cinematic lighting, vibrant colors, clear background, professional AI-powered story video, 3D diagrams and labels, 4K resolution, 16:9 aspect ratio",
            },
            with_logs=True,
        )
        print(f"Image generation result: {result}")
        if 'images' in result and len(result['images']) > 0 and 'url' in result['images'][0]:
            return result['images'][0]['url']
        return None
    except Exception as e:
        print(f"Error in image generation: {str(e)}")
        return None


def __generate_google_veo_video(prompt, image_url=None):
    """Generates a high-quality video using Google Veo 3.1 (Lumiere's successor) on fal.ai"""
    print(f"Generating Google Veo 3.1 video for prompt: {prompt[:100]}...")
    try:
        if image_url:
            print(f"Using Image-to-Video with Google Veo 3.1...")
            result = fal_client.subscribe(
                "fal-ai/veo3.1/image-to-video",
                arguments={
                    "image_url": image_url,
                    "prompt": f"{prompt}. Google Lumiere style, highly detailed 3D visualization, realistic motion, professional educational 3D animation, cinematic lighting, 4K resolution.",
                    "resolution": "1080p",
                    "aspect_ratio": "16:9",
                    "generate_audio": True
                },
                with_logs=True,
            )
        else:
            print(f"Using Text-to-Video with Google Veo 3.1...")
            result = fal_client.subscribe(
                "fal-ai/veo3.1",
                arguments={
                    "prompt": f"{prompt}. Google Lumiere style, highly detailed 3D visualization, realistic motion, professional educational 3D animation, cinematic lighting, 4K resolution.",
                    "resolution": "1080p",
                    "aspect_ratio": "16:9",
                    "generate_audio": True
                },
                with_logs=True,
            )
        
        print(f"Google Veo result: {result}")
        if 'video' in result and 'url' in result['video']:
            return result['video']['url']
        elif 'url' in result:
            return result['url']
        return None
    except Exception as e:
        print(f"Error in Google Veo video generation: {str(e)}")
        return None


def __animate_image(image_url):
    """Animates an image using fal-ai/luma-dream-machine/ray-2/image-to-video (Lumiere-like lifelike motion)"""
    print(f"Animating image: {image_url}")
    try:
        result = fal_client.subscribe(
            "fal-ai/luma-dream-machine/ray-2/image-to-video",
            arguments={
                "image_url": image_url,
                "prompt": "Lumiere style lifelike motion, cinematic 3D animation, fluid character movement, professional educational storytelling, 3D animated objects and environments, smooth camera pans and zooms, high-quality rendering, vibrant colors, clear transitions, educational 3D visual effects",
            },
            with_logs=True,
        )
        print(f"Animation result: {result}")
        if 'video' in result and 'url' in result['video']:
            return result['video']['url']
        return None
    except Exception as e:
        print(f"Error in image animation: {str(e)}")
        return None


def generate_story(image_file, prompt, genre, style, language, word_count, creativity):
    """Generates a story given an image or prompt"""
    # Ensure access counts are initialized
    if app_config.openai_curr_access_count is None:
        mongo.fetch_curr_access_count()
        
    max_count = app_config.openai_max_access_count
    curr_count = app_config.openai_curr_access_count
    available_count = max_count - curr_count

    # Check if image is an LFS pointer
    if image_file and __is_lfs_pointer(image_file):
        return "ERROR: The selected example image is a Git LFS pointer and not a real image file. Please upload your own image or pull the full LFS files from the repository.", max_count, curr_count, available_count
        
    image_desc = ""
    if image_file:
        # read image as bytes array
        with open(image_file, "rb") as f:
            input_image = f.read()
        # generate caption for image
        image_desc = __image2text(image=input_image)
    else:
        # If no image is provided, use the prompt to generate an image context
        image_desc = prompt if prompt else "A scene for a story"

    # generate story from caption
    story = __text2story(
        image_desc=image_desc,
        prompt=prompt,
        genre=genre,
        style=style,
        language=language,
        word_count=word_count,
        creativity=creativity,
    )
    # Clean the story script
    story = __clean_script(story)
    
    # increment the openai access counter and compute count stats
    mongo.increment_curr_access_count()
    curr_count = app_config.openai_curr_access_count
    available_count = max_count - curr_count
    return story, max_count, curr_count, available_count


def __text2audio(text):
    """Generates an audio from text using fal-ai/chatterbox/text-to-speech"""
    print("Generating audio from text...")
    try:
        result = fal_client.subscribe(
            "fal-ai/chatterbox/text-to-speech",
            arguments={
                "text": text,
            },
            with_logs=True,
        )
        print(f"Audio raw result: {result}")
        
        if 'audio' in result and 'url' in result['audio']:
            return result['audio']['url']
        elif 'url' in result:
            return result['url']
        else:
            print(f"Warning: Unexpected audio generation result format: {result}")
            return None
    except Exception as e:
        print(f"Error in audio generation: {str(e)}")
        return None


def __merge_video_audio(video_url, audio_url):
    """Merges video and audio using fal-ai/ffmpeg-api/merge-audio-video"""
    print("Merging video and audio...")
    try:
        result = fal_client.subscribe(
            "fal-ai/ffmpeg-api/merge-audio-video",
            arguments={
                "video_url": video_url,
                "audio_url": audio_url,
            },
            with_logs=True,
        )
        print(f"Merge raw result: {result}")
        
        if 'video' in result and 'url' in result['video']:
            return result['video']['url']
        elif 'url' in result:
            return result['url']
        else:
            print(f"Warning: Unexpected merge result format: {result}")
            return video_url
    except Exception as e:
        print(f"Error in video-audio merging: {str(e)}")
        return video_url # Return original video if merge fails


def generate_video(source_file, prompt, genre, style, language, word_count, creativity, engine="Google Veo"):
    """Generates a voiced video from image or prompt using fal.ai (Google Veo or Luma) or locally"""
    print(f"Starting voiced video generation with {engine} for {source_file if source_file else 'prompt only'}")
    
    image_file = source_file

    # Check if image is an LFS pointer
    if image_file and __is_lfs_pointer(image_file):
        print("ERROR: Selected image is an LFS pointer.")
        return "ERROR: LFS pointer image selected.", None, 0, 0, 0
        
    # Ensure access counts are initialized
    if app_config.openai_curr_access_count is None:
        mongo.fetch_curr_access_count()
        
    max_count = app_config.openai_max_access_count
    curr_count = app_config.openai_curr_access_count
    available_count = max_count - curr_count

    if not os.getenv("FAL_KEY"):
        print("ERROR: FAL_KEY not found in environment.")
        # Fallback to local if FAL_KEY is missing, but only if we have an image
        if not image_file:
             return "ERROR: FAL_KEY not found and no image provided.", None, image_file, max_count, curr_count, available_count

    try:
        image_url = None
        image_desc = ""
        
        if image_file:
            # Generate caption for the image
            with open(image_file, "rb") as f:
                input_image = f.read()
            image_desc = __image2text(image=input_image)
            
            # 1. Prepare image for fal.ai using data URI
            if os.getenv("FAL_KEY"):
                try:
                    print(f"Step 1: Preparing image {image_file} as data URI...")
                    with open(image_file, "rb") as f:
                        img_data = f.read()
                    import base64
                    base64_img = base64.b64encode(img_data).decode("utf-8")
                    image_url = f"data:image/jpeg;base64,{base64_img}"
                except Exception as e:
                    print(f"Failed to prepare data URI: {str(e)}")
        else:
            # Generate image from prompt
            if not prompt:
                raise Exception("Either an image or a prompt is required.")
            if os.getenv("FAL_KEY"):
                try:
                    print(f"Step 1: Generating image from prompt: {prompt}")
                    image_url = __text2image(prompt)
                except Exception as e:
                    print(f"Image generation failed: {str(e)}")
            image_desc = prompt

        if image_desc.startswith("ERROR:"):
            raise Exception(f"Image captioning failed: {image_desc}")
        
        # 2. Generate specialized script
        if genre == "3D Visualization" or style == "3D Visualization":
             video_explanation_prompt = (
                f"SYSTEM: You are a professional 3D scientific and technical visualization expert. Create a 60-second script for a high-quality 3D visualization video about: {prompt if prompt else image_desc}.\n"
                f"STYLE: Photorealistic 3D visualization, detailed internal views, exploded diagrams, transparent materials, and cinematic camera movements.\n"
                f"OUTPUT FORMAT:\n"
                f"Title: (visualization title)\n"
                f"Narration: (Write the clear, technical yet accessible 60-second narration script here including a strong conclusion beginning with 'Conclusion: ')"
            )
        elif genre == "Internal Process" or style == "Internal Process":
            video_explanation_prompt = (
                f"SYSTEM: You are an expert in industrial and biological processes. Create a 60-second script for an 'Internal Process' video that shows the inner workings of: {prompt if prompt else image_desc}.\n"
                f"STYLE: Macro 3D animation, 'X-ray' style views, fluid dynamics, and molecular/mechanical precision.\n"
                f"OUTPUT FORMAT:\n"
                f"Title: (process title)\n"
                f"Narration: (Write the step-by-step 60-second narration script here including a strong conclusion beginning with 'Conclusion: ')"
            )
        else:
            video_explanation_prompt = (
                f"SYSTEM: You are a professional 3D animator and educator. Create an engaging 60-second Lumiere-style animation script with lifelike motion explaining the topic: {prompt if prompt else image_desc}.\n"
                f"STYLE: Professional 3D Pixar-style educational animation, highly detailed 3D characters and environment, cinematic lighting, and realistic, fluid lifelike movement.\n"
                f"OUTPUT FORMAT:\n"
                f"Title: (creative educational title)\n"
                f"Narration: (Write the clear, engaging 60-second narration script here including a strong conclusion beginning with 'Conclusion: ')"
            )

        story = __text2story(
            image_desc=image_desc,
            prompt=video_explanation_prompt,
            genre=genre,
            style=style,
            language=language,
            word_count=word_count,
            creativity=creativity
        )
        story = __clean_script(story)
        print(f"Generated script: {story}")
        
        if story.startswith("ERROR:") or story.startswith("Cannot generate story"):
            raise Exception(f"Story generation failed: {story}")

        # 3. Generate Video
        if os.getenv("FAL_KEY"):
            print(f"Step 3: Creating AI video with {engine}...")
            if engine == "Google Veo":
                final_video_url = __generate_google_veo_video(story, image_url)
            else:
                final_video_url = __animate_image(image_url)
                # For Luma, we still need to generate audio and merge
                audio_url = __text2audio(story)
                if audio_url:
                    final_video_url = __merge_video_audio(final_video_url, audio_url)
            
            if not final_video_url:
                print(f"{engine} failed, falling back to local denoising video...")
                final_video_path = __create_denoising_video(image_file, story, "output_video.mp4", language=language)
            else:
                final_video_path = final_video_url
        else:
            # Fallback to local denoising
            img_path = image_file if image_file else "assets/examples/teacher-school.jpg"
            final_video_path = __create_denoising_video(img_path, story, "output_video.mp4", language=language)
        
        print(f"Final Video generated: {final_video_path}")
        
        # increment the access counter
        mongo.increment_curr_access_count()
        curr_count = app_config.openai_curr_access_count
        available_count = max_count - curr_count
        
        return story, final_video_path, (image_file if image_file else image_url), max_count, curr_count, available_count
        
    except Exception as e:
        print(f"Error in voiced video generation: {str(e)}")
        return f"Error: {str(e)}", None, source_file, max_count, curr_count, available_count
