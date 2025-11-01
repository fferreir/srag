import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 1. Inicializar o App Flask
app = Flask(__name__)

# 2. Carregar o pipeline (apenas uma vez, quando o app inicia)
model_path = "modelo_lgbm.joblib"
try:
    pipeline = joblib.load(model_path)
    print("Pipeline carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    pipeline = None

# 3. Definir um endpoint de "health check"
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# 4. Definir o endpoint de predição
@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({"error": "Modelo não carregado"}), 500

    try:
        # 1. Pegar os dados JSON da requisição
        data = request.get_json(force=True)

        # 2. Converter o JSON para um DataFrame do Pandas
        # Esperamos um JSON no formato de 'colunas' ou 'dicionário de linhas'
        # Ex: { "col_a": [1, 2], "col_b": ["x", "y"] }
        # ou: [ { "col_a": 1, "col_b": "x" }, { "col_a": 2, "col_b": "y" } ]

        # Este formato é mais flexível:
        if isinstance(data, list):
            predict_df = pd.DataFrame.from_records(data)
        elif isinstance(data, dict):
             # Se for um único registro: { "col_a": 1, "col_b": "x" }
             # ou múltiplos registros: { "col_a": [1, 2], "col_b": ["x", "y"] }
            predict_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Formato JSON não suportado"}), 400

        # 3. Usar o pipeline para pré-processar E prever
        # O pipeline lida com dados "crus"
        probabilidades = pipeline.predict_proba(predict_df)

        # 4. Formatar a resposta
        # Vamos retornar a probabilidade da classe '1' (positiva)
        output = list(probabilidades[:, 1])

        return jsonify({"predictions_prob_class_1": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 5. Rodar o app (apenas para debug local, NÃO para produção)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
