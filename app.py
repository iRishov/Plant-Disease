from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import os
from keras.utils import load_img, img_to_array # type: ignore


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from the .keras file


MODEL_PATH = r"G:\Trained Models\InceptionResNetV2_model.keras"
if os.path.exists(MODEL_PATH):
    MODEL = tf.keras.models.load_model(MODEL_PATH, compile=True) # type: ignore
else:
    MODEL = None
    print(f"⚠ Warning: Model file {MODEL_PATH} not found!")

# Updated class labels for 38 classes (PlantVillage dataset)
LABELS = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___Healthy",
    4: "Blueberry___Healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___Healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot",
    8: "Corn_(maize)___Common_rust",
    9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___Healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___Healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___Healthy",
    18: "Pepper__bell___Bacterial_spot",
    19: "Pepper__bell___Healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___Healthy",
    23: "Raspberry___Healthy",
    24: "Soybean___Healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___Healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites_Two_spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Yellow_Leaf__Curl_Virus",
    36: "Tomato___Mosaic_virus",
    37: "Tomato___Healthy"
}

# Updated disease details for each class (placeholders—update with actual details)
DISEASE_DETAILS = {
    "Apple___Apple_scab": {
        "symptoms": "Dark, scabby lesions on leaves and fruits.",
        "disease_cycle": "Fungal spores overwinter in fallen leaves and debris.",
        "pesticide_usage": "Fungicides may be required during wet weather."
    },
    "Apple___Black_rot": {
        "symptoms": "Dark lesions on leaves, fruits, and twigs.",
        "disease_cycle": "Fungus survives on infected plant debris.",
        "pesticide_usage": "Copper-based sprays or specific fungicides."
    },
    "Apple___Cedar_apple_rust": {
        "symptoms": "Orange pustules on leaves and fruit; leaf spots.",
        "disease_cycle": "Requires both apple and juniper hosts to complete its cycle.",
        "pesticide_usage": "Fungicide applications during early infection stages."
    },
    "Apple___Healthy": {
        "symptoms": "No disease symptoms; normal appearance.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Blueberry___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "symptoms": "White powdery growth on leaves and shoots.",
        "disease_cycle": "Fungus overwinters on bud scales and twigs.",
        "pesticide_usage": "Sulfur-based fungicides are commonly used."
    },
    "Cherry_(including_sour)___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Corn_(maize)___Cercospora_leaf_spot": {
        "symptoms": "Small, circular spots with gray centers on leaves.",
        "disease_cycle": "Pathogen overwinters in crop residue.",
        "pesticide_usage": "Fungicides and crop rotation are recommended."
    },
    "Corn_(maize)___Common_rust": {
        "symptoms": "Rust-colored pustules on leaf surfaces.",
        "disease_cycle": "Fungus overwinters in alternate hosts and crop residue.",
        "pesticide_usage": "Timely fungicide application is essential."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms": "Long, elliptical lesions with a tan center.",
        "disease_cycle": "Pathogen overwinters in corn residue.",
        "pesticide_usage": "Use of fungicides and resistant varieties."
    },
    "Corn_(maize)___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Grape___Black_rot": {
        "symptoms": "Small, brown spots that enlarge and become corky.",
        "disease_cycle": "Fungus overwinters in mummified berries and twigs.",
        "pesticide_usage": "Regular fungicide sprays are needed during wet periods."
    },
    "Grape___Esca_(Black_Measles)": {
        "symptoms": "Brownish discoloration and necrotic spots on leaves.",
        "disease_cycle": "Fungal pathogens colonize the vascular tissues.",
        "pesticide_usage": "Management is difficult; cultural practices help."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms": "Spotted leaves with yellow halos.",
        "disease_cycle": "Fungus survives on fallen leaves.",
        "pesticide_usage": "Fungicide applications are recommended."
    },
    "Grape___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": "Yellow shoots, blotchy mottling on leaves, and misshapen fruits.",
        "disease_cycle": "Bacterial infection spread by the Asian citrus psyllid.",
        "pesticide_usage": "Management includes vector control and removal of infected trees."
    },
    "Peach___Bacterial_spot": {
        "symptoms": "Dark, water-soaked lesions on leaves and fruits.",
        "disease_cycle": "Bacteria overwinter in twigs and buds.",
        "pesticide_usage": "Copper sprays and proper sanitation are recommended."
    },
    "Peach___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Pepper__bell___Bacterial_spot": {
        "symptoms": "Small, water-soaked lesions that turn brown on leaves.",
        "disease_cycle": "Bacteria spread through splashing water and contaminated seeds.",
        "pesticide_usage": "Copper-based sprays are typically used."
    },
    "Pepper__bell___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Potato___Early_blight": {
        "symptoms": "Small, dark spots on older leaves with concentric rings.",
        "disease_cycle": "Fungus survives on plant debris and in the soil.",
        "pesticide_usage": "Fungicides and crop rotation are key management strategies."
    },
    "Potato___Late_blight": {
        "symptoms": "Large, water-soaked lesions with white mold on the undersides of leaves.",
        "disease_cycle": "Pathogen thrives in cool, wet conditions.",
        "pesticide_usage": "Fungicides and field sanitation are critical."
    },
    "Potato___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Raspberry___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Soybean___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Squash___Powdery_mildew": {
        "symptoms": "White, powdery spots on leaves and stems.",
        "disease_cycle": "Fungus overwinters in plant debris and spreads in dry conditions.",
        "pesticide_usage": "Sulfur-based fungicides are often applied."
    },
    "Strawberry___Leaf_scorch": {
        "symptoms": "Brownish lesions on leaves with scorched margins.",
        "disease_cycle": "Disease often results from environmental stress combined with pathogen infection.",
        "pesticide_usage": "Fungicide treatment may be required in severe cases."
    },
    "Strawberry___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    },
    "Tomato___Bacterial_spot": {
        "symptoms": "Small, dark spots on leaves and fruits.",
        "disease_cycle": "Bacteria spread by splashing water and contaminated seeds.",
        "pesticide_usage": "Copper-based sprays and resistant varieties help manage the disease."
    },
    "Tomato___Early_blight": {
        "symptoms": "Dark spots with concentric rings on lower leaves.",
        "disease_cycle": "Fungal spores survive in plant debris and soil.",
        "pesticide_usage": "Fungicides such as Chlorothalonil and Mancozeb are effective."
    },
    "Tomato___Late_blight": {
        "symptoms": "Large, water-soaked lesions with white mold underneath.",
        "disease_cycle": "Pathogen thrives in humid, cool conditions.",
        "pesticide_usage": "Timely fungicide application is crucial."
    },
    "Tomato___Leaf_mold": {
        "symptoms": "Yellow spots on upper leaf surfaces and mold on undersides.",
        "disease_cycle": "Fungus develops in warm, humid conditions.",
        "pesticide_usage": "Fungicides and proper ventilation can help manage the disease."
    },
    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Small, circular spots with gray centers on leaves.",
        "disease_cycle": "Pathogen overwinters in infected plant debris.",
        "pesticide_usage": "Regular fungicide sprays and removal of infected leaves are recommended."
    },
    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "symptoms": "Fine webbing and stippling on leaves, leading to discoloration.",
        "disease_cycle": "Mites multiply rapidly in hot, dry conditions.",
        "pesticide_usage": "Miticides or natural remedies like neem oil are effective."
    },
    "Tomato___Target_Spot": {
        "symptoms": "Dark, circular lesions with concentric rings on leaves.",
        "disease_cycle": "Fungal spores are spread by wind and rain.",
        "pesticide_usage": "Fungicides and removal of infected plant parts are advised."
    },
    "Tomato___Yellow_Leaf__Curl_Virus": {
        "symptoms": "Yellowing and upward curling of leaves with stunted growth.",
        "disease_cycle": "Virus is transmitted by whiteflies in warm climates.",
        "pesticide_usage": "Vector control and resistant varieties are key."
    },
    "Tomato___Mosaic_virus": {
        "symptoms": "Mottled appearance on leaves, stunted growth, and deformed fruits.",
        "disease_cycle": "Virus is spread by contact and infected seeds.",
        "pesticide_usage": "No chemical cure exists; removal of infected plants is essential."
    },
    "Tomato___Healthy": {
        "symptoms": "No visible disease symptoms.",
        "disease_cycle": "",
        "pesticide_usage": ""
    }
}

def preprocess_image(file_bytes, target_size=(224, 224)):
      
    img = load_img(BytesIO(file_bytes), target_size=target_size)    
    input_arr = img_to_array(img)   
    input_arr = np.array([input_arr])
    return input_arr


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model is not available. Please check the model path."}
    
    try:
        file_bytes = await file.read()
        image_array = preprocess_image(file_bytes)
        predictions = MODEL.predict(image_array)[0]
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

