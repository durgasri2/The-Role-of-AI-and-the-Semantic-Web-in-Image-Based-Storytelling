import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    title = "AI Story Teller"
    theme = "freddyaboulton/dracula_revamped"
    css = "style.css"
    openai_max_access_count = 200
    openai_curr_access_count = None
    mongo_client = None
    db = "mydb"
    collection = "pic2story-openai-access-counter"
    key = "current_count"
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MONGO_CONN_STR = os.getenv("MONGO_CONN_STR")
    I2T_API_URL = os.getenv("I2T_API_URL")
    HF_TOKEN = os.getenv("HF_TOKEN")
    genre_list = genre = [
        "3D Animation",
        "3D Visualization",
        "Internal Process",
        "AI Story Teller",
        "Adventure",
        "Children Literature",
        "Comedy",
        "Drama",
        "Fantasy",
        "Fiction",
        "Horror",
        "Mystery",
        "Non-fiction",
        "Poetry",
        "Romance",
        "Satire",
        "Surrealism",
        "Urban Fantasy",
    ]
    writing_style_list = [
        "3D Animation",
        "3D Visualization",
        "Internal Process",
        "AI Story Teller",
        "Cinematic",
        "Conversational",
        "Descriptive",
        "Experimental",
        "First-Person",
        "Formal",
        "Informal",
        "Metaphorical",
        "Minimalist",
        "Narrative",
        "Non-linear",
        "Objective",
        "Sensory",
        "Stream of Consciousness",
        "Symbolic",
        "Third-Person Limited",
        "Third-Person Omniscient",
    ]
    language_list = [
        "English",
        "Indian English",
        "Hindi",
        "Telugu",
    ]


app_config = AppConfig()
