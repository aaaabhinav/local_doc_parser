from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.schemas.responses import ExtractionResponse
from app.services.extraction import ExtractionService
from app.utils.logger import logger
import time

extract_router = APIRouter()

@extract_router.post("/extract", response_model=ExtractionResponse)
async def extract_document(request: Request, file: UploadFile = File(...)):
    start_time = time.time()
    
    # Very basic content-type check
    if not file.content_type.startswith("image/") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image or PDF.")

    logger.info(f"Received file upload: {file.filename} ({file.content_type})")
    
    file_bytes = await file.read()
    
    llm_engine = request.app.state.llm
    ocr_engine = request.app.state.ocr
    
    service = ExtractionService(llm_engine, ocr_engine)
    
    try:
        result, confidence = await service.process_document(file_bytes)
        
        process_time_ms = int((time.time() - start_time) * 1000)
        overall_conf = sum(confidence.values()) / len(confidence) if confidence else 0.0
        
        return ExtractionResponse(
            success=True,
            document_type=result.get("document_type", "unknown"),
            data=result,
            field_confidence=confidence,
            overall_confidence=overall_conf,
            processing_time_ms=process_time_ms
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return ExtractionResponse(
            success=False,
            document_type="unknown",
            data={},
            field_confidence={},
            overall_confidence=0.0,
            processing_time_ms=int((time.time() - start_time) * 1000),
            error=str(e)
        )
