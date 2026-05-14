from pydantic import BaseModel
from typing import Dict, Any, Optional

class ExtractionResponse(BaseModel):
    success: bool
    document_type: str
    data: Dict[str, Any]
    field_confidence: Dict[str, float]
    overall_confidence: float
    processing_time_ms: int
    error: Optional[str] = None
