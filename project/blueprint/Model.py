import uuid
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required
from marshmallow import fields
from webargs.flaskparser import use_args
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from http import HTTPStatus
from project.app.decorator import admin_required

bp = Blueprint('model', __name__)

model_path = os.path.join(os.path.dirname(__file__), "trained_model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}.")

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_cures = {
    "Apple___Apple_scab": "Apply fungicides like Captan, Myclobutanil, or Mancozeb at early stages to prevent spore germination. Prune infected leaves and ensure good airflow between trees to reduce humidity. Use disease-resistant apple varieties such as Liberty, Freedom, or Enterprise. Practice crop rotation and remove fallen leaves and debris to break the disease cycle. Consider applying neem oil or potassium bicarbonate as organic alternatives.",
    "Apple___Black_rot": "Remove and destroy infected branches, leaves, and fruits to prevent fungal spread. Apply copper-based fungicides during the growing season, especially after periods of heavy rain. Improve air circulation by proper pruning and avoid excessive nitrogen fertilization, which can promote disease susceptibility. Clean up fallen leaves and debris, as the fungus overwinters in mummified fruit. Maintain orchard sanitation by regularly disinfecting pruning tools.",
    "Apple___Cedar_apple_rust": "Use fungicides containing Myclobutanil or Propiconazole early in the season before symptoms appear. Plant resistant apple varieties such as Redfree, Liberty, or Enterprise to reduce infection risk. Remove nearby cedar trees that may host the rust spores or prune galls from them in late winter. Apply sulfur-based organic treatments to protect young foliage. Avoid overhead watering to minimize moisture levels on leaves.",
    "Apple___healthy": "No treatment needed. Ensure proper pruning, watering, and fertilization for strong tree growth. Monitor for pests and diseases regularly to catch any issues early. Mulch around the base of trees to retain soil moisture and suppress weed growth. Encourage beneficial insects like ladybugs and lacewings to naturally control pests.",
    "Cherry_(including_sour)___Powdery_mildew": "Spray sulfur-based fungicides or potassium bicarbonate as soon as white powdery patches appear. Prune infected areas and increase air circulation around trees to reduce humidity. Apply neem oil or a mix of milk and water (1:9 ratio) as organic treatment options. Avoid overhead watering, as excess moisture encourages fungal growth. Plant resistant varieties and maintain proper spacing between trees to limit disease spread.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides like Azoxystrobin, Propiconazole, or Mancozeb before symptoms become severe. Rotate crops with non-host plants like soybeans or legumes to disrupt the disease cycle. Remove and destroy infected plant residue after harvest to prevent overwintering spores. Improve soil drainage and avoid excessive nitrogen fertilization, which can promote disease severity. Use resistant corn hybrids for long-term disease management.",
    "Corn_(maize)___Common_rust_": "Plant resistant corn varieties to minimize disease susceptibility. Ensure proper spacing between plants to promote air circulation and reduce humidity. If infection spreads widely, apply Mancozeb or Copper-based fungicides. Monitor crops closely and remove any infected leaves to limit fungal reproduction. Rotate with non-host crops like wheat or soybeans to break the fungal lifecycle.",
    "Grape___Black_rot": "Prune and remove all infected leaves, fruits, and vines during dormancy to reduce fungal spores. Apply fungicides such as Mancozeb, Myclobutanil, or Captan at bud break and repeat as needed. Maintain proper canopy management by training vines to allow sunlight penetration and air circulation. Avoid wetting the leaves while watering, and remove any fallen fruit or leaves from the vineyard. Consider using organic treatments like copper-based sprays for sustainable disease management.",
    "Grape___Esca_(Black_Measles)": "Avoid excessive irrigation, as high moisture levels promote fungal growth. Remove infected vine wood by pruning during the dormant season and apply wound protectants to prevent reinfection. Use balanced fertilization with adequate potassium to strengthen vines against disease stress. Apply Trichoderma-based biological fungicides to suppress Esca pathogens in the soil. Monitor for early symptoms like leaf discoloration and intervene promptly.",
    "Orange___Haunglongbing_(Citrus_greening)": "Unfortunately, there is no cure for this disease once a tree is infected. Control Asian citrus psyllid populations with insecticides like Imidacloprid or Spinosad. Use reflective mulches to deter psyllid insects from landing on trees. Regularly remove and destroy infected trees to prevent disease spread. Support plant health with balanced fertilization and proper irrigation.",
    "Potato___Early_blight": "Apply fungicides like Chlorothalonil or Copper-based solutions before lesions become widespread. Rotate crops to avoid planting potatoes in the same soil consecutively, which can increase disease risk. Water at the base of the plants to minimize leaf wetness and fungal spread. Use mulch to prevent soil splash, which carries fungal spores. Plant resistant potato varieties such as Defender or Elba.",
    "Potato___Late_blight": "Use Copper-based fungicides or Mancozeb at the first sign of infection. Immediately remove and destroy infected plants to prevent rapid disease spread. Avoid overhead irrigation and ensure proper soil drainage to reduce excess moisture. Space plants adequately to allow airflow, reducing humidity levels. Choose blight-resistant potato varieties such as Sarpo Mira.",
    "Tomato___Bacterial_spot": "Spray copper-based fungicides weekly, especially in wet conditions. Avoid handling plants when they are wet to prevent bacterial spread. Use disease-free seeds and rotate crops yearly to reduce bacterial buildup in the soil. Increase air circulation around plants by pruning lower leaves. Soak seeds in a 10% bleach solution before planting to eliminate potential bacterial contamination.",
    "Tomato___Early_blight": "Apply Chlorothalonil-based fungicides at the first sign of symptoms. Remove infected leaves immediately and avoid composting them. Keep foliage dry by watering at the base of the plant instead of overhead. Rotate crops and avoid planting tomatoes in the same location year after year. Plant resistant tomato varieties like Mountain Magic or Iron Lady.",
    "Tomato___Late_blight": "Use Copper-based fungicides or Chlorothalonil sprays to slow disease progression. Remove and destroy infected plants to prevent spores from spreading to nearby plants. Increase spacing between plants and avoid excessive humidity in greenhouses. Monitor for symptoms frequently and act quickly to remove infected areas. Grow blight-resistant varieties such as Mountain Merit.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "No direct cure exists for this viral disease. Control whiteflies using insecticidal soap, neem oil, or sticky traps. Use virus-resistant tomato varieties like Tygress or Shanty. Remove and destroy infected plants immediately to limit virus spread. Install insect netting around tomato fields to prevent whitefly infestations.",
    "Tomato___Tomato_mosaic_virus": "There is no treatment for this virus, but preventative measures can limit its spread. Remove and destroy infected plants as soon as symptoms appear. Sterilize gardening tools with a bleach solution after handling infected plants. Use resistant tomato varieties such as Celebrity or Big Beef. Avoid tobacco exposure, as the virus can spread from infected cigarettes or hands."
}

def preprocess_image(image_file):
    image = Image.open(io.BytesIO(image_file.read()))
    image = image.convert("RGB")
    image = image.resize((260, 194))
    jpg_path = f"{uuid.uuid4().hex}.jpg"
    image.save(jpg_path, "JPEG")
    return jpg_path

@bp.route('/api/image', methods=['POST'])
@use_args({"image": fields.Field(required=True)}, location='files')
@jwt_required()
# @admin_required
def predict_plant_disease(args):
    image_file = args['image']
    processed_image_path = preprocess_image(image_file)
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(processed_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    if result_index >= len(class_names):
        return jsonify({"error": "prediction result out of range!"}), HTTPStatus.NOT_FOUND
    predicted_result = class_names[result_index]
    cure_suggestions = disease_cures.get(predicted_result)
    return jsonify({'plant name' : predicted_result,'predicted_results': cure_suggestions}), HTTPStatus.OK