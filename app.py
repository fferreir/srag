import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
from ml_utils import convert_to_string


# 1. Inicializar o App Flask
app = Flask(__name__)

# 2. Carregar o pipeline (agora deve funcionar)
model_path = "modelo_lgbm.joblib"
try:
    pipeline = joblib.load(model_path)
    print("Pipeline carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}") # O erro real aparecerá aqui
    pipeline = None

# ... (o resto do seu app.py, que está correto) ...
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({"error": "Modelo não carregado"}), 500

    try:
        data = request.get_json(force=True)

        if isinstance(data, list):
            predict_df = pd.DataFrame.from_records(data)
        elif isinstance(data, dict):
            predict_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Formato JSON não suportado"}), 400

        probabilidades = pipeline.predict_proba(predict_df)
        output = list(probabilidades[:, 1])

        return jsonify({"predictions_prob_class_1": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
