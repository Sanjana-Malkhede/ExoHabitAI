from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from backend.utils import predict_habitability
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="/static"
)

CORS(app)


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = predict_habitability(data)
        
        habitability_label = (
            "Habitable"
            if result["prediction"] == 1
            else "Not Habitable"
        )

        return jsonify({
            "status": "success",
            "prediction": {
                "habitability": habitability_label,
                "score": result["probability"]
            }
        })

    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Internal server error: " + str(e)
        }), 500


@app.route("/rank", methods=["POST"])
def rank():
    try:
        data = request.get_json()

        if not isinstance(data, list):
            return jsonify({"error": "Expected list of exoplanets"}), 400

        df = pd.DataFrame(data)

        from backend.utils import model, FEATURE_COLUMNS

        df_model = df[FEATURE_COLUMNS]
        df["habitability_probability"] = model.predict_proba(df_model)[:, 1]

        ranked = df.sort_values(
            by="habitability_probability",
            ascending=False
        )

        return jsonify({
            "status": "success",
            "results": ranked.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
