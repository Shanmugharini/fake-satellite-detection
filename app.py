from flask import Flask, request, render_template
import os
import numpy as np
import cv2
import pickle
from sklearn.decomposition import PCA
from skimage.filters import gabor
import pywt

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model and PCA
def load_model():
    try:
        with open("xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("pca_model.pkl", "rb") as f:
            pca = pickle.load(f)
        print(" Models loaded successfully.")
        return model, pca
    except Exception as e:
        print(" Error loading models:", e)
        raise e

# Feature extraction
class CustomPixelHop:
    def __init__(self, num_kernels=10):
        self.pca = PCA(n_components=num_kernels)

    def fit_transform(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.pca.fit_transform(X_flat)

def gabor_features(img, frequency=0.3):
    features = []
    for i in range(3):
        for theta in (0, np.pi / 3, 2 * np.pi / 3):
            real, _ = gabor(img[:, :, i], frequency=frequency, theta=theta)
            features.append(real.mean())
    return np.array(features)

def wavelet_features(img):
    features = []
    for i in range(3):
        coeffs = pywt.wavedec2(img[:, :, i], 'haar', level=2)
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                for c in coeff:
                    features.append(np.mean(c))
                    features.append(np.std(c))
            else:
                features.append(np.mean(coeff))
                features.append(np.std(coeff))
    return np.array(features)

def extract_features_from_image(img):
    try:
        img = cv2.resize(img, (256, 256))
        img = img / 255.0

        patch_size = (16, 16)
        img_patches = [img[i:i+patch_size[0], j:j+patch_size[1], :]
                    for i in range(0, img.shape[0], patch_size[0])
                    for j in range(0, img.shape[1], patch_size[1])]
        img_patches = np.array(img_patches).reshape(-1, patch_size[0], patch_size[1], 3)

        pixelhop1 = CustomPixelHop(10)
        ph1 = pixelhop1.fit_transform(img_patches)

        pixelhop2 = CustomPixelHop(10)
        ph2 = pixelhop2.fit_transform(ph1)

        pixelhop_feats = np.array([ph1.mean(), ph2.mean()])
        gabor_feats = gabor_features(img)
        wavelet_feats = wavelet_features(img)

        combined_features = np.hstack((pixelhop_feats, gabor_feats, wavelet_feats))
        print("Features extracted.")
        return combined_features
    except Exception as e:
        print("Feature extraction failed:", e)
        raise e

def predict_image(img_path):
    try:
        print(f" Reading image from: {img_path}")
        model, pca = load_model()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image could not be read. Invalid format or corrupt file.")

        features = extract_features_from_image(img)
        features_pca = pca.transform([features])
        prediction = model.predict(features_pca)[0]
        print(f" Prediction done: {prediction}")
        return prediction
    except Exception as e:
        print(" Prediction failed:", e)
        raise e

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return render_template('index.html', prediction="No image uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        os.makedirs('static', exist_ok=True)
        filepath = os.path.abspath(os.path.join('static', file.filename))
        file.save(filepath)

        result = predict_image(filepath)
        label = "Fake Satellite Image" if result == 1 else "Real Satellite Image"
        return render_template('index.html', prediction=label, image_path=filepath)

    except Exception as e:
        print("Error in /predict route:", e)
        return render_template('index.html', prediction="Prediction failed. Try again.")

if __name__ == '__main__':
    app.run(debug=True)
