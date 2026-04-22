import os
import logging
from typing import Optional, Tuple
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv

set_llm_cache(InMemoryCache())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

FLASK_ENV = os.getenv('FLASK_ENV', 'development')

app = Flask(__name__)
CORS(app)

if FLASK_ENV == 'production':
    app.config['ENV'] = 'production'
    app.config['DEBUG'] = False
    logging.getLogger().setLevel(logging.INFO)
else:
    app.config['ENV'] = 'development'
    app.config['DEBUG'] = True
    logging.getLogger().setLevel(logging.DEBUG)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = os.path.join('models', 'MobileNetV2.h5')
TARGET_IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Food classification labels
CLASS_NAMES = [
    'ayam_goreng', 'ayam_pop', 'daging_rendang', 'dendeng_batokok', 'gulai_ikan', 
    'gulai_tambusu', 'gulai_tunjang', 'telur_balado', 'telur_dadar'
]

def initialize_langchain_gemini():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        
        # Initiate LLM via LongChain
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        logger.info("LangChain Gemini AI initialized")
        return llm
    except Exception as e:
        logger.error(f"LangChain Init Error: {str(e)}")
        return None

def load_classification_model() -> Optional[tf.keras.Model]:
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at: {MODEL_PATH}")
            return None
            
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Classification model successfully loaded from: {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from '{MODEL_PATH}': {str(e)}")
        return None

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> Optional[np.ndarray]:
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        logger.debug(f"Image preprocessed successfully: {image_path}")
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image '{image_path}': {str(e)}")
        return None

recipe_template = """
Generate a cooking recipe for Padang (Indonesian) Cuisine food: "{food_name}".

STRICT RULES:
- Write ONLY the recipe content in Bahasa Indonesia.
- Start directly with the recipe title.
- Format using Markdown with EXACT structure:
  ### Deskripsi
  ### Bahan-bahan
  ### Cara Membuat

LANGUAGE: Bahasa Indonesia.
"""
prompt_template = PromptTemplate.from_template(recipe_template)

image_classifier_model = load_classification_model()
llm_chain = prompt_template | initialize_langchain_gemini()

def generate_recipe_langchain(food_name: str) -> str:
    try:
        # Cache Checking
        response = llm_chain.invoke({"food_name": food_name})
        return response.content
    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        return "Gagal generate resep, coba lagi nanti."


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API status."""
    status = {
        'status': 'healthy',
        'classifier_model': image_classifier_model is not None,
        'recipe_generation': llm_chain is not None
    }
    return jsonify(status), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expected request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: file (image file)
    """
    
    if image_classifier_model is None:
        logger.error("Prediction request failed: Model not available")
        return jsonify({
            'error': 'Classification model is not available on the server'
        }), 503


    if 'file' not in request.files:
        logger.warning("Prediction request missing file")
        return jsonify({
            'error': 'No file provided in the request'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Prediction request with empty filename")
        return jsonify({
            'error': 'No file selected for upload'
        }), 400
    

    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        return jsonify({
            'error': f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File uploaded successfully: {filename}")

        # Preprocess image
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            return jsonify({
                'error': 'Invalid image format or corrupted file'
            }), 400

        # Perform prediction
        prediction = image_classifier_model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        food_name_raw = CLASS_NAMES[predicted_class_idx]
        food_name_display = food_name_raw.replace('_', ' ').title()
        
        logger.info(
            f"Prediction completed: {food_name_display} "
            f"(confidence: {confidence:.2%})"
        )

        # Generate recipe
        recipe_text = generate_recipe_langchain(food_name_display)


        base_url = request.host_url.rstrip('/')
        if '0.0.0.0' in base_url:
            base_url = base_url.replace('0.0.0.0', '127.0.0.1')
        image_url = f"{base_url}/uploads/{filename}"

        return jsonify({
        'success': True,
        'food_name': food_name_display,
        'confidence': round(confidence, 4),
        'recipe': recipe_text,
        'image_url': image_url
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred during processing'
        }), 500


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file '{filename}': {str(e)}")
        return jsonify({'error': 'File not found'}), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size too large error."""
    return jsonify({
        'error': 'File size exceeds maximum limit (16MB)'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error'
    }), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)