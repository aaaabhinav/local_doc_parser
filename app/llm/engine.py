import os
from llama_cpp import Llama
from app.utils.logger import logger

class LLMEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        
    def load(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Please download it.")
            return

        logger.info(f"Loading LLM from {self.model_path}...")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1, # Changed to -1 to use your NVIDIA GPU
                n_ctx=4096,
                n_threads=8,
                verbose=False
            )
            logger.info("LLM Engine successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")

    def generate(self, prompt: str) -> str:
        if not self.llm:
            logger.warning("LLM is not loaded. Using mock JSON response to test the pipeline flow.")
            return """```json
{
  "document_type": "warranty_card",
  "productName": "AXIOM X1 Pro",
  "brand": "AXIOM",
  "purchaseDate": "2024-05-15",
  "warrantyDuration": "1Y",
  "tags": ["smartphone", "electronics"],
  "purchasePrice": {
    "currency": "INR",
    "amount": 24999
  },
  "serialNumber": "AX1P256BL24051567",
  "hasFreeServiceVisits": true,
  "freeServices": [
    {
      "serviceName": "General Check-up & Cleaning",
      "serviceDate": "2024-11-15",
      "reminderWeeksBefore": 1
    },
    {
      "serviceName": "Screen & Touch Check",
      "serviceDate": "2025-04-15",
      "reminderWeeksBefore": 1
    }
  ]
}
```"""
            
        logger.info("Running LLM inference...")
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a specialized document extraction AI. You only output strict JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
        )
        return response["choices"][0]["message"]["content"]
