import os
import io
import uuid
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
from supabase import create_client, Client
from dotenv import load_dotenv
import base64


load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/images'
app.config['ALLOWED_EXTENSIONS'] = {'h5', 'hdf5', 'png', 'jpg', 'jpeg', 'jfif', 'webp'}

# Ensure image directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_adversarial_pattern(input_image, input_label, model, epsilon):
    input_image = tf.convert_to_tensor(input_image)
    input_label = tf.convert_to_tensor(input_label)
    
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.MSE(input_label, prediction)
    
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    adversarial_image = input_image + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # keep it 32x32
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def image_to_base64(img_array, upscale_to=(256, 256)):
    if isinstance(img_array, tf.Tensor):
        img_array = img_array.numpy()

    img_array = (img_array * 255).astype('uint8')

    # Get first image in batch if necessary
    if img_array.ndim == 4:
        img_array = img_array[0]

    img = Image.fromarray(img_array)

    # Upscale for better visibility
    img = img.resize(upscale_to, Image.NEAREST)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Handle single image in batch
    if len(img_array.shape) == 4:
        img_array = img_array[0]
    
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    try:
        response = supabase.table("CustomModel").select("id, name, fileType, fileSize, createdAt").execute()
        models = response.data
        return render_template("index.html", models=models)
    except Exception as e:
        return jsonify({"error": "Failed to fetch models", "details": str(e)}), 500

def pgd_attack(image, label, model, epsilon=0.3, alpha=0.01, iters=40):
    """Projected Gradient Descent attack"""
    adv_image = tf.identity(image)
    for i in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = tf.keras.losses.categorical_crossentropy(label, prediction)
        gradient = tape.gradient(loss, adv_image)
        adv_image = adv_image + alpha * tf.sign(gradient)
        perturbation = tf.clip_by_value(adv_image - image, -epsilon, epsilon)
        adv_image = tf.clip_by_value(image + perturbation, 0, 1)
    return adv_image

def bim_attack(image, label, model, epsilon=0.3, alpha=0.01, iters=10):
    """Basic Iterative Method attack"""
    adv_image = tf.identity(image)
    for i in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = tf.keras.losses.categorical_crossentropy(label, prediction)
        gradient = tape.gradient(loss, adv_image)
        adv_image = adv_image + alpha * tf.sign(gradient)
        adv_image = tf.clip_by_value(adv_image, image - epsilon, image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image


@app.route('/bulk_classify', methods=['POST'])
def bulk_classify():
    try:
        # Get form data
        images = request.files.getlist("images")
        model_id = request.form.get("model_id")
        attack_type = request.form.get("attack_type", "none")
        epsilon = float(request.form.get("epsilon", 0.05))
        alpha = float(request.form.get("alpha", 0.01))
        iterations = int(request.form.get("iterations", 10))

        if not images or not model_id:
            return jsonify({"error": "Missing image files or model ID"}), 400

        # Fetch model from Supabase with error handling
        try:
            response = supabase.table("CustomModel").select("fileData").eq("id", model_id).execute()
            if not response.data:
                return jsonify({"error": "Model not found"}), 404
            
            model_row = response.data[0]  # Get first item instead of using .single()
            file_data = model_row.get("fileData")

            if not file_data or not file_data.startswith("\\x"):
                return jsonify({"error": "Invalid model data format"}), 400

            # Save model to temporary file
            model_bytes = bytes.fromhex(file_data[2:])
            model_path = os.path.join("uploads/models", f"{uuid.uuid4().hex}_bulk.h5")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, "wb") as f:
                f.write(model_bytes)

            model = load_model(model_path)

            # Initialize counters
            total_original_conf, total_adv_conf, flip_count = 0, 0, 0
            total_images = len(images)
            processed_images = 0

            for img_file in images:
                try:
                    # Load and process image
                    img = Image.open(img_file).convert('RGB').resize((32, 32))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    # Original prediction
                    original_pred = model.predict(img_array, verbose=0)
                    original_class = np.argmax(original_pred[0])
                    original_conf = float(original_pred[0][original_class])

                    label = np.zeros((1, 10))  # Assuming 10 classes - adjust as needed
                    label[0, original_class] = 1

                    # Generate adversarial image
                    if attack_type == 'fgsm':
                        adversarial_image = generate_adversarial_pattern(img_array, label, model, epsilon)
                    elif attack_type == 'pgd':
                        adversarial_image = pgd_attack(
                            tf.convert_to_tensor(img_array),
                            tf.convert_to_tensor(label),
                            model,
                            epsilon=epsilon,
                            alpha=alpha,
                            iters=iterations
                        )
                    elif attack_type == 'bim':
                        adversarial_image = bim_attack(
                            tf.convert_to_tensor(img_array),
                            tf.convert_to_tensor(label),
                            model,
                            epsilon=epsilon,
                            alpha=alpha,
                            iters=iterations
                        )
                    else:
                        adversarial_image = img_array  # no attack

                    # Adversarial prediction
                    adv_pred = model.predict(adversarial_image, verbose=0)
                    adv_class = np.argmax(adv_pred[0])
                    adv_conf = float(adv_pred[0][original_class])  # Confidence in original class

                    # Update counters
                    total_original_conf += original_conf
                    total_adv_conf += adv_conf
                    if original_class != adv_class:
                        flip_count += 1

                    processed_images += 1

                except Exception as e:
                    print(f"Error processing image {img_file.filename}: {str(e)}")
                    continue

            # Calculate averages
            if processed_images == 0:
                return jsonify({"error": "No images processed successfully"}), 400

            avg_original_conf = total_original_conf / processed_images
            avg_adv_conf = total_adv_conf / processed_images
            avg_drop = avg_original_conf - avg_adv_conf
            flip_percent = (flip_count / processed_images) * 100

            return jsonify({
                "success": True,
                "total_images": total_images,
                "processed_images": processed_images,
                "avg_original_confidence": round(avg_original_conf * 100, 2),
                "avg_adversarial_confidence": round(avg_adv_conf * 100, 2),
                "avg_confidence_drop": round(avg_drop * 100, 2),
                "flip_percent": round(flip_percent, 2),
                "attack_type": attack_type
            })

        finally:
            # Clean up model file
            if 'model_path' in locals() and os.path.exists(model_path):
                os.remove(model_path)

    except Exception as e:
        return jsonify({
            "error": "An error occurred during processing",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)