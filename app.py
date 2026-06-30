import gradio as gr
import model
from config import app_config
import mongo_utils as mongo
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

 
def clear():
    return (
        None,
        "",
        app_config.default_genre,
        app_config.default_style,
        app_config.default_language,
        app_config.default_word_count,
        app_config.default_creativity,
        app_config.default_engine,
        "",
        None,
    )


def create_interface():
    js_enable_darkmode = """() => 
    {
        document.querySelector('body').classList.add('dark');
    }"""
    js_toggle_darkmode = """() => 
    {
        if (document.querySelectorAll('.dark').length) {
            document.querySelector('body').classList.remove('dark');
        } else {
            document.querySelector('body').classList.add('dark');
        }
    }"""

    with gr.Blocks(title=app_config.title, theme=app_config.theme, css=app_config.css) as app:
        # enable darkmode
        app.load(fn=None, inputs=None, outputs=None, js=js_enable_darkmode)
        with gr.Row():
            darkmode_checkbox = gr.Checkbox(
                label="Dark Mode", value=True, interactive=True
            )
            # toggle darkmode on/off when checkbox is checked/unchecked
            darkmode_checkbox.change(
                None, None, None, js=js_toggle_darkmode, api_name=False
            )
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    # AI Story Sphere
                    """
                )
                image = gr.Image(
                    type="filepath",
                    label="Upload Image"
                )
                prompt = gr.Textbox(
                    label="User Prompt (Optional):",
                    placeholder="Enter additional instructions for the story...",
                )
                with gr.Row():
                    with gr.Column():
                        genre = gr.Dropdown(
                            label="Story Genre: ",
                            value="3D Animation",
                            choices=app_config.genre,
                        )
                        style = gr.Dropdown(
                            label="Story Writing Style:",
                            value="3D Animation",
                            choices=app_config.writing_style_list,
                        )
                        language = gr.Dropdown(
                            label="Story Language:",
                            value="English",
                            choices=app_config.language_list,
                        )
                    with gr.Column():
                        # Word Count Slider
                        word_count = gr.Slider(
                            label="Story Length (words):",
                            minimum=30,
                            maximum=200,
                            value=50,
                            step=10,
                        )
                        creativity = gr.Slider(
                            label="Creativity Index:",
                            minimum=0.3,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                        )
                with gr.Row():
                    engine = gr.Dropdown(
                        label="AI Video Engine:",
                        value="Google Veo",
                        choices=["Google Veo", "Luma Ray 2"],
                    )
                with gr.Row():
                    submit_button = gr.Button(
                        value="Generate Story", elem_classes="orange-button"
                    )
                    video_button = gr.Button(
                        value="Generate Video", elem_classes="orange-button"
                    )
                    clear_button = gr.ClearButton(elem_classes="gray-button")
            with gr.Column():
                max_count = gr.Textbox(
                    label="Max allowed API requests:",
                    value=app_config.openai_max_access_count,
                )
                curr_count = gr.Textbox(
                    label="Used up API requests:",
                    value=app_config.openai_curr_access_count,
                )
                available_count = gr.Textbox(
                    label="Available API requests:",
                    value=app_config.openai_max_access_count
                    - app_config.openai_curr_access_count,
                )
                story = gr.Textbox(
                    label="Story Text:",
                    placeholder="Generated story will appear here.",
                    lines=10,
                )
                video_output = gr.Video(label="Story Video:")
        submit_button.click(
            fn=model.generate_story,
            inputs=[image, prompt, genre, style, language, word_count, creativity],
            outputs=[story, max_count, curr_count, available_count],
        )

        video_button.click(
            fn=model.generate_video,
            inputs=[image, prompt, genre, style, language, word_count, creativity, engine],
            outputs=[story, video_output, image, max_count, curr_count, available_count],
        )

        clear_button.click(
            fn=clear, inputs=[], outputs=[image, prompt, genre, style, language, word_count, creativity, engine, story, video_output]
        )
        image.clear(fn=clear, inputs=[], outputs=[image, prompt, genre, style, language, word_count, creativity, engine, story, video_output])
    return app


if __name__ == "__main__":
    print("Fetching access count...")
    mongo.fetch_curr_access_count()
    print("Creating interface...")
    app_gradio = create_interface()
    
    # Create FastAPI app
    app_fastapi = FastAPI()
    
    # Serve the login page at root
    @app_fastapi.get("/")
    async def read_index():
        return FileResponse(os.path.join("login", "index.html"))
    
    # Mount static files for the login page
    app_fastapi.mount("/login", StaticFiles(directory="login"), name="login")
    
    # Mount Gradio app to /app
    app_fastapi = gr.mount_gradio_app(app_fastapi, app_gradio, path="/app")
    
    print("Launching app on http://127.0.0.1:7860")
    uvicorn.run(app_fastapi, host="127.0.0.1", port=7860)
