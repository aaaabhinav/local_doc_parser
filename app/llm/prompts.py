def build_extraction_prompt(ocr_text: str) -> str:
    return f"""
You are a highly capable document data extraction assistant.
Analyze the following OCR text extracted from a document.
Determine the document_type (receipt, invoice, warranty_card, service_record, note).
Extract all relevant fields into the EXACT JSON format below to match our Product Details form.
If a field is not found in the text, leave it as an empty string "" (or null for numbers/booleans). DO NOT guess or hallucinate values.

For `warrantyDuration`, pick one of: "None", "6M", "1Y", "2Y", "3Y", "5Y", or "Custom".
For `tags`, extract a list of descriptive words like "smartphone", "electronics", etc.
For `purchasePrice`, extract currency ("INR", "USD", etc.) and amount as a number.
For `freeServices`, this should be an array. If there are 0 services, return `[]`. If there are 2 services, return 2 objects in the array. Extract EVERY service listed.

OCR TEXT:
----------------
{ocr_text}
----------------

REQUIRED JSON OUTPUT FORMAT (output ONLY valid JSON, nothing else):
{{
  "document_type": "warranty_card",
  "productName": "Product Name",
  "brand": "Brand Name",
  "purchaseDate": "YYYY-MM-DD",
  "warrantyDuration": "1Y",
  "tags": ["electronics"],
  "purchasePrice": {{
    "currency": "INR",
    "amount": 24999
  }},
  "serialNumber": "AX1P256BL24051567",
  "hasFreeServiceVisits": true,
  "freeServices": [
    {{
      "serviceName": "General Check-up",
      "serviceDate": "YYYY-MM-DD",
      "reminderWeeksBefore": 1
    }}
  ]
}}
"""
