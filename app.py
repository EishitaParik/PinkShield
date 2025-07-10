import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask

from flask import Flask, request, render_template, send_file
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from datetime import datetime

# Load model and scaler
model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

app = Flask(__name__)

def create_pdf_report(features, prediction, confidence, shap_image_path):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y_pos = height - 50

    # Report title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, y_pos, "Breast Cancer Prediction Report")

    # Metadata: date/time & report ID
    y_pos -= 40
    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now().strftime("%d %B %Y %I:%M %p")
    c.setFont("Helvetica", 10)
    c.drawString(50, y_pos, f"Date: {now}")
    c.drawString(400, y_pos, f"Report ID: #{report_id}")

    # Input Features
    y_pos -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "Input Features:")
    y_pos -= 15

    c.setFont("Helvetica", 10)
    feature_lines = []
    current_line = ""
    for i, val in enumerate(features):
        entry = f"{float(val):.2f}"
        if len(current_line) + len(entry) < 90:
            current_line += entry + ", "
        else:
            feature_lines.append(current_line)
            current_line = entry + ", "
    feature_lines.append(current_line)

    for line in feature_lines:
        c.drawString(50, y_pos, line.strip(', '))
        y_pos -= 15

    # Prediction result
    y_pos -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, f"Prediction Result: {prediction}")

    # Confidence
    y_pos -= 20
    c.setFont("Helvetica", 11)
    c.drawString(50, y_pos, f"Model Confidence: {confidence:.2f}")

    # Advice
    y_pos -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "Advice:")
    y_pos -= 15
    c.setFont("Helvetica", 10)
    advice = "Please consult a medical professional for diagnosis and treatment." if prediction == "cancerous" else \
             "No cancer detected. Maintain regular checkups and a healthy lifestyle."
    c.drawString(50, y_pos, advice)

    # SHAP Image
    y_pos -= 100
    if os.path.exists(shap_image_path):
        try:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_pos + 80, "Top 5 Contributing Features (SHAP):")
            shap_img = ImageReader(shap_image_path)
            c.drawImage(shap_img, 50, y_pos - 150, width=500, height=150, preserveAspectRatio=True)
            y_pos -= 180
        except Exception as e:
            print("❌ Error adding SHAP image to PDF:", e)

    # Disclaimer
    y_pos -= 30
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    disclaimer = "This is an AI-generated prediction and not a medical diagnosis."
    c.drawString(50, y_pos, disclaimer)

    c.save()
    buffer.seek(0)
    return buffer

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    features = request.form.get('feature', '')
    features_lst = [f.strip() for f in features.split(',') if f.strip()]

    if len(features_lst) != 30:
        return render_template('index.html', message=["❌ Please enter exactly 30 comma-separated values!"])

    try:
        np_features = np.asarray(features_lst, dtype=np.float32).reshape(1, -1)
    except ValueError:
        return render_template('index.html', message=["❌ Please enter valid numbers only!"])

    scaled_input = scaler.transform(np_features)
    pred = model.predict(scaled_input)
    proba = model.predict_proba(scaled_input)[0][1]  # Probability of being cancerous

    output = "cancerous" if pred[0] == 1 else "not cancerous"

    # SHAP plot
    explainer = shap.Explainer(model, masker=shap.maskers.Independent(scaled_input))
    shap_values = explainer(scaled_input)
    plt.clf()
    shap.plots.bar(shap_values[0], max_display=5, show=False)
    shap_img_id = f"shap_{uuid.uuid4().hex}.png"
    shap_img_path = os.path.join('static', shap_img_id)
    plt.savefig(shap_img_path, bbox_inches='tight')

    # Store shap_img_id temporarily (for HTML display and reuse in PDF)
    return render_template('index.html', message=[output], shap_img=shap_img_id, proba=proba, features=features)

@app.route("/download_report", methods=["POST"])
def download_report():
    raw_input = request.form.get('feature', '')
    shap_img = request.form.get('shap_img', None)
    features = [f.strip() for f in raw_input.split(',') if f.strip()]

    if len(features) != 30:
        return "Invalid input features", 400

    try:
        features_float = [float(f) for f in features]
    except Exception as e:
        print("❌ Error converting features:", e)
        return "Invalid feature values", 400

    np_features = np.asarray(features_float, dtype=np.float32).reshape(1, -1)
    scaled_input = scaler.transform(np_features)
    pred = model.predict(scaled_input)
    proba = model.predict_proba(scaled_input)[0][1]
    output = "cancerous" if pred[0] == 1 else "not cancerous"

    shap_img_path = os.path.join('static', shap_img) if shap_img else None
    pdf_buffer = create_pdf_report(features_float, output, proba, shap_img_path)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="breast_cancer_prediction_report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)