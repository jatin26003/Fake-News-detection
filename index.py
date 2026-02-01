from flask import Flask, request, render_template_string
import pickle
import os

app = Flask(__name__)

# Load model safely (serverless-friendly)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "fake_news_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECT_PATH, "rb"))

HTML = """ 
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap" rel="stylesheet">

    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 900px;
            margin: 60px auto;
            background: white;
            padding: 40px;
            border-radius: 18px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.25);
            animation: fadeIn 0.8s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            text-align: center;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            height: 170px;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #ccc;
            font-size: 15px;
            resize: none;
        }

        .actions {
            text-align: center;
            margin-top: 20px;
        }

        button {
            background: #667eea;
            color: white;
            padding: 14px 30px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            background: #5a67d8;
            transform: scale(1.03);
        }

        .examples {
            text-align: center;
            margin-top: 15px;
        }

        .examples button {
            background: #e2e8f0;
            color: #333;
            margin: 5px;
            padding: 8px 14px;
            font-size: 13px;
        }

        .result-box {
            margin-top: 35px;
            padding: 25px;
            border-radius: 14px;
            background: #f8f9fc;
        }

        .real {
            color: #28a745;
            font-weight: 700;
        }

        .fake {
            color: #dc3545;
            font-weight: 700;
        }

        .progress {
            height: 12px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }

        .progress-bar {
            height: 100%;
            border-radius: 10px;
        }

        .loader {
            display: none;
            text-align: center;
            margin-top: 20px;
        }

        .footer {
            text-align: center;
            font-size: 12px;
            color: #777;
            margin-top: 30px;
        }

        @media (max-width: 600px) {
            .container {
                margin: 20px;
                padding: 25px;
            }
        }
    </style>

    <script>
        function showLoader() {
            document.getElementById("loader").style.display = "block";
        }

        function fillExample(text) {
            document.getElementById("news").value = text;
        }
    </script>
</head>

<body>

<div class="container">
    <h1>📰 Fake News Detection</h1>
    <p class="subtitle">AI-powered system to identify misleading or false news content</p>

    <form method="post" onsubmit="showLoader()">
        <textarea id="news" name="news" placeholder="Paste the news article text here..." required></textarea>

        <div class="actions">
            <button type="submit">Analyze News</button>
        </div>
    </form>

    <div class="examples">
        <p><b>Try an example:</b></p>
        <button onclick="fillExample('“The United Nations said on Tuesday that global food prices declined slightly in June, driven by lower cereal and vegetable oil costs, according to a report released by the Food and Agriculture Organization.”')">
            Sample REAL
        </button>
        <button onclick="fillExample('Shocking secret revealed! Government hiding truth from citizens!!!')">
            Sample FAKE
        </button>
    </div>

    <div id="loader" class="loader">
        🔄 Analyzing news using AI...
    </div>

    {% if result %}
    <div class="result-box">
        <h2>
            Prediction:
            {% if result == "REAL" %}
                <span class="real">✔ REAL</span>
            {% else %}
                <span class="fake">✖ FAKE</span>
            {% endif %}
        </h2>

        <p>Confidence Level: <b>{{ confidence }}%</b></p>

        <div class="progress">
            <div class="progress-bar" style="width: {{ confidence }}%; background: {% if result == 'REAL' %}#28a745{% else %}#dc3545{% endif %};"></div>
        </div>
    </div>
    {% endif %}

    <div class="footer">
        🔒 Your data is processed locally and not stored
    </div>
</div>

</body>
</html>
"""  # (paste your polished UI HTML here unchanged)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        news = request.form.get("news", "")
        vec = vectorizer.transform([news])
        prob = model.predict_proba(vec)[0]

        real_prob = prob[1]
        if real_prob >= 0.6:
            result = "REAL"
        elif real_prob <= 0.4:
            result = "FAKE"
        else:
            result = "UNCERTAIN"

        confidence = round(max(prob) * 100, 2)

    return render_template_string(HTML, result=result, confidence=confidence)

# IMPORTANT: export the app (DO NOT use app.run)
