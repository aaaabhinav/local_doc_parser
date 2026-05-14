from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Local AI Extraction Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Path to your GGUF model
    # For now, pointing to a local models folder. You will need to download the .gguf file.
    MODEL_PATH: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    class Config:
        env_file = ".env"

settings = Settings()
