from typing import Dict
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TrOCR processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')


# Define endpoint to run the model
@app.post("/run")
def run_model(data: Dict[str, str]):
    # Load image from the provided URL
    url = data.get('url')
    if not url:
        return {"error": "URL not provided"}

    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Process image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"generated_text": generated_text}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port=8000)