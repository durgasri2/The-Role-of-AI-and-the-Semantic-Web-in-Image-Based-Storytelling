# Repository Guidelines

## Project Structure & Module Organization
The project is an AI-powered story generator that transforms images into narrative text. It uses a hybrid architecture combining **Gradio** for the UI and **FastAPI** for serving.

- **`app.py`**: The main entry point. It defines the Gradio interface, integrates it into a FastAPI application, and serves a static login page from the `login/` directory.
- **`model.py`**: Contains the core logic for the AI pipeline. It handles image captioning (via Groq Vision or Hugging Face BLIP) and story generation (via LangChain with OpenAI or Groq).
- **`config.py`**: Manages application settings and environment variables (via `python-dotenv`). It defines the `AppConfig` dataclass for UI themes, API keys, and model parameters.
- **`mongo_utils.py`**: Provides utilities for MongoDB interaction, primarily used for tracking and limiting API access counts.
- **`login/`**: A self-contained directory with HTML, CSS, and JS for the application's landing/login page.

## Build, Test, and Development Commands
The application is built with Python and relies on external AI APIs.

- **Install dependencies**: `pip install -r requirements.txt`
- **Run the application**: `python app.py` (Launches the server at `http://127.0.0.1:7860`)
- **Main App URL**: The Gradio interface is mounted at `/app`, while the root `/` serves the login page.

## Coding Style & Naming Conventions
- **Naming**: Uses `snake_case` for functions and variables, and `PascalCase` for classes (e.g., `AppConfig`).
- **Environment**: Configuration is strictly managed through a `.env` file. Key variables include `OPENAI_KEY`, `GROQ_API_KEY`, `MONGO_CONN_STR`, `I2T_API_URL`, and `HF_TOKEN`.
- **Modularity**: Logic is separated into functional modules (`model.py` for AI, `mongo_utils.py` for DB, `config.py` for settings).

## Integration & Dependencies
- **UI**: Gradio `3.46.0`.
- **AI Orchestration**: LangChain (OpenAI/Groq).
- **Database**: MongoDB (via `pymongo`).
