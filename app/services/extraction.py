import json
import asyncio
import time
from app.utils.logger import logger
from app.llm.prompts import build_extraction_prompt

class ExtractionService:
    def __init__(self, llm_engine, ocr_engine):
        self.llm = llm_engine
        self.ocr = ocr_engine

    async def process_document(self, file_bytes: bytes):
        logger.info("Starting document processing pipeline...")
        pipeline_start = time.time()
        
        # 1. OCR Extraction (Run in threadpool to prevent blocking)
        ocr_start = time.time()
        ocr_text, ocr_conf = await asyncio.to_thread(self.ocr.extract, file_bytes)
        ocr_duration = time.time() - ocr_start
        
        if not ocr_text.strip():
            raise ValueError("No text detected in image")
            
        logger.info(f"[TIMING] OCR completed in {ocr_duration:.2f} seconds. Extracted {len(ocr_text)} characters.")

        # 2. LLM Processing
        llm_start = time.time()
        prompt = build_extraction_prompt(ocr_text)
        llm_response = await asyncio.to_thread(self.llm.generate, prompt)
        llm_duration = time.time() - llm_start
        logger.info(f"[TIMING] LLM inference completed in {llm_duration:.2f} seconds.")
        
        # 3. JSON Repair & Parsing
        extracted_data = self._parse_json(llm_response)
        
        # 4. Confidence Calculation (Mock field confidence for now)
        field_confidence = {
            k: ocr_conf.get("ocr_mean_confidence", 0.85) for k in extracted_data.keys()
        }
        
        total_duration = time.time() - pipeline_start
        
        print("\n" + "="*50)
        print(f"⏱️ TIMING REPORT:")
        print(f"  -> OCR Processing: {ocr_duration:.2f} seconds")
        print(f"  -> LLM Processing: {llm_duration:.2f} seconds")
        print(f"  -> Total Pipeline: {total_duration:.2f} seconds")
        print("="*50 + "\n")
        
        logger.info(f"[TIMING] Document processing pipeline complete. Total time: {total_duration:.2f} seconds.")
        return extracted_data, field_confidence
        
    def _parse_json(self, text: str) -> dict:
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nRaw output: {text}")
            return {"error": "Failed to extract structured data", "raw_text": text}
