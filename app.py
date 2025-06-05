from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import os
from keras.utils import load_img, img_to_array # type: ignore
from disease_data import LABELS, DISEASE_DETAILS # Import labels and details


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model paths
MODEL_PATHS = {
    "cnn": r"G:\Trained Models\cnn.keras",
    "efficientnetv2s": r"G:\Trained Models\EfficientNetV2S_model.keras",
    "inceptionresnetv2": r"G:\Trained Models\InceptionResNetV2_model.keras",
}

# Cache for loaded models
LOADED_MODELS = {}

# Updated class labels for 38 classes (PlantVillage dataset)
# LABELS = { ... moved to disease_data.py }

# Updated disease details for each class (placeholders—update with actual details)
# DISEASE_DETAILS = { ... moved to disease_data.py }

# Helper function to load model
def load_model(model_name: str):
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        print(f"⚠ Warning: Model name {model_name} not found in MODEL_PATHS!")
        return None

    if not os.path.exists(model_path):
        print(f"⚠ Warning: Model file {model_path} not found!")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=True) # type: ignore
        LOADED_MODELS[model_name] = model # Cache the loaded model
        print(f"✓ Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"❌ Error loading model {model_name} from {model_path}: {e}")
        return None

# Preload all models at startup
@app.on_event("startup")
async def startup_event():
    print("Starting to preload all models...")
    for model_name in MODEL_PATHS.keys():
        load_model(model_name)
    print("All models preloaded successfully!")

def preprocess_image(file_bytes, target_size=(224, 224)):
      
    img = load_img(BytesIO(file_bytes), target_size=target_size)    
    input_arr = img_to_array(img)   
    input_arr = np.array([input_arr])
    return input_arr


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Query("efficientnetv2s")):
    # Default to efficientnetv2s if no model_name is provided
    model = load_model(model_name)

    if model is None:
        return {"error": f"Model '{model_name}' is not available or could not be loaded. Please check the model name and path."}
    
    try:
        file_bytes = await file.read()
        image_array = preprocess_image(file_bytes)
        predictions = model.predict(image_array)[0]
        confidence = float(np.max(predictions))
        
        if confidence < 0.4:  # Adjust confidence threshold if needed
            return {"error": "Unknown Image - Model is not confident enough."}
        
        predicted_label = LABELS[np.argmax(predictions)] # type: ignore
        disease_details = DISEASE_DETAILS.get(predicted_label, {})
        return {"class": predicted_label, "confidence": confidence, "details": disease_details}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

