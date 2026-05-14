from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import extract_router
from app.core.config import settings
from app.llm.engine import LLMEngine
from app.ocr.engine import OCREngine
from app.utils.logger import logger
import os

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Enable CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Local AI Extraction Backend...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Initialize engines
    app.state.ocr = OCREngine()
    
    app.state.llm = LLMEngine(model_path=settings.MODEL_PATH)
    app.state.llm.load()

app.include_router(extract_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {"message": "Local AI Extraction Backend API is running."}
