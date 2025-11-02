#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import datetime
import dateutil
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    make_scorer
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
import matplotlib.pyplot as plt
import joblib


dados = pd.read_parquet('influenza_ML_2025-10-30_16-28-12.parquet')

dados.drop(['DT_SIN_PRI', 'OBES_IMC', 'FEBRE', 'TABAG'], axis=1, inplace=True)

dados.drop(dados.loc[dados['UTI'] == 9].index, axis=0, inplace=True)

dados.drop(dados.loc[dados['UTI'].isnull()].index, axis=0, inplace=True)

dados = dados[['CS_SEXO',
 'CS_GESTANT',
 'CS_RACA',
 'CS_ESCOL_N',
 'ID_PAIS',
 'CS_ZONA',
 'NOSOCOMIAL',
 'AVE_SUINO',
 'TOSSE',
 'GARGANTA',
 'DISPNEIA',
 'DESC_RESP',
 'SATURACAO',
 'DIARREIA',
 'VOMITO',
 'PUERPERA',
 'CARDIOPATI',
 'HEMATOLOGI',
 'SIND_DOWN',
 'HEPATICA',
 'ASMA',
 'DIABETES',
 'NEUROLOGIC',
 'PNEUMOPATI',
 'IMUNODEPRE',
 'RENAL',
 'OBESIDADE',
 'VACINA',
 'MAE_VAC',
 'M_AMAMENTA',
 'ANTIVIRAL',
 'TP_ANTIVIR',
 'HOSPITAL',
 'UTI',
 'RAIOX_RES',
 'PCR_RESUL',
 'POS_PCRFLU',
 'TP_FLU_PCR',
 'PCR_FLUASU',
 'PCR_FLUBLI',
 'POS_PCROUT',
 'PCR_VSR',
 'PCR_PARA1',
 'PCR_PARA2',
 'PCR_PARA3',
 'PCR_PARA4',
 'PCR_ADENO',
 'PCR_METAP',
 'PCR_BOCA',
 'PCR_RINO',
 'PCR_OUTRO',
 'DOR_ABD',
 'FADIGA',
 'PERD_OLFT',
 'PERD_PALA',
 'TOMO_RES',
 'TP_TES_AN',
 'RES_AN',
 'POS_AN_FLU',
 'TP_FLU_AN',
 'POS_AN_OUT',
 'AN_SARS2',
 'AN_VSR',
 'AN_PARA1',
 'AN_PARA2',
 'AN_PARA3',
 'AN_ADENO',
 'AN_OUTRO',
 'POV_CT',
 'SURTO_SG',
 'IDADE',
 'DIAS_UL_VAC',
 'DIAS_UL_VAC_MAE',
 'DIAS_UL_VAC_DOSEUNI',
 'DIAS_UL_VAC_1_DOSE',
 'DIAS_UL_VAC_2_DOSE',
 'DIAS_UL_INIC_ANTIVIRAL',
 'DIAS_INTERNACAO',
 'DIAS_INTERNA_RX_RESP',
 'DIAS_SINT_INI_RX_RESP',
 'DIAS_INTERNA_TOMO',
 'DIAS_SINT_INI_TOMO'
]]

y = dados['UTI']

y = y.map({1.0: 1, 2.0: 0})

X = dados.drop('UTI', axis=1)

colunas_categoricas = [ 'CS_SEXO',
 'CS_GESTANT',
 'CS_RACA',
 'CS_ESCOL_N',
 'ID_PAIS',
 'CS_ZONA',
 'NOSOCOMIAL',
 'AVE_SUINO',
 'TOSSE',
 'GARGANTA',
 'DISPNEIA',
 'DESC_RESP',
 'SATURACAO',
 'DIARREIA',
 'VOMITO',
 'PUERPERA',
 'CARDIOPATI',
 'HEMATOLOGI',
 'SIND_DOWN',
 'HEPATICA',
 'ASMA',
 'DIABETES',
 'NEUROLOGIC',
 'PNEUMOPATI',
 'IMUNODEPRE',
 'RENAL',
 'OBESIDADE',
 'VACINA',
 'MAE_VAC',
 'M_AMAMENTA',
 'ANTIVIRAL',
 'TP_ANTIVIR',
 'HOSPITAL',
 'RAIOX_RES',
 'PCR_RESUL',
 'POS_PCRFLU',
 'TP_FLU_PCR',
 'PCR_FLUASU',
 'PCR_FLUBLI',
 'POS_PCROUT',
 'PCR_VSR',
 'PCR_PARA1',
 'PCR_PARA2',
 'PCR_PARA3',
 'PCR_PARA4',
 'PCR_ADENO',
 'PCR_METAP',
 'PCR_BOCA',
 'PCR_RINO',
 'PCR_OUTRO',
 'DOR_ABD',
 'FADIGA',
 'PERD_OLFT',
 'PERD_PALA',
 'TOMO_RES',
 'TP_TES_AN',
 'RES_AN',
 'POS_AN_FLU',
 'TP_FLU_AN',
 'POS_AN_OUT',
 'AN_SARS2',
 'AN_VSR',
 'AN_PARA1',
 'AN_PARA2',
 'AN_PARA3',
 'AN_ADENO',
 'AN_OUTRO',
 'POV_CT',
 'SURTO_SG']

colunas_quantitativas = ['IDADE',
 'DIAS_UL_VAC',
 'DIAS_UL_VAC_MAE',
 'DIAS_UL_VAC_DOSEUNI',
 'DIAS_UL_VAC_1_DOSE',
 'DIAS_UL_VAC_2_DOSE',
 'DIAS_UL_INIC_ANTIVIRAL',
 'DIAS_INTERNACAO',
 'DIAS_INTERNA_RX_RESP',
 'DIAS_SINT_INI_RX_RESP',
 'DIAS_INTERNA_TOMO',
 'DIAS_SINT_INI_TOMO']

def convert_to_string(x):
    """Converte a entrada para o tipo string, para o pipeline."""
    return x.astype(str)

# ---
# Bloco 2: DEFINIÇÃO DO PIPELINE
# ---
print("\nDefinindo o pré-processamento e o pipeline...")

# 1. Transformador numérico: Preenche NaNs com mediana E escala
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Transformador categórico:
#    - Preenche NaNs com a string "AUSENTE"
#    - Converte TUDO para string (para lidar com colunas mistas de float/str)
#    - Codifica strings para inteiros (trata categorias novas como -1)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='AUSENTE')),
    ('to_string', FunctionTransformer(convert_to_string, feature_names_out='one-to-one')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int))
])

# 3. Juntar os transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, colunas_quantitativas),
        ('cat', categorical_transformer, colunas_categoricas)
    ],
    remainder='passthrough' # Mantém colunas não listadas (se houver)
)

# 4. Identificar índices das features categóricas APÓS a transformação
# (Importante para o LightGBM saber quais colunas tratar como categóricas)
start_cat_index = len(colunas_quantitativas)
end_cat_index = start_cat_index + len(colunas_categoricas)
categorical_feature_indices = list(range(start_cat_index, end_cat_index))

print(f"Índices Categóricos para o LGBM: {categorical_feature_indices}")

# ---
# Bloco 3: Divisão de Treino e Teste
# ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nDados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# ---
# Bloco 4: Função 'Objective' do Optuna (Adaptada para Pipeline)
# ---
# Esta função processa os dados internamente para cada 'trial'
def objective(trial):
    # 1. Definir o espaço de busca
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0)
    }

    # Usamos n_estimators alto, pois o early_stopping cuidará disso
    model = lgb.LGBMClassifier(**params, n_estimators=2000)

    # 2. Recriar o preprocessor para este 'trial'
    # (Usamos os pipelines já definidos: numeric_transformer, categorical_transformer)
    preprocessor_obj = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, colunas_quantitativas),
            ('cat', categorical_transformer, colunas_categoricas)
        ],
        remainder='passthrough'
    )

    # 3. Criar conjunto de validação interna
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # 4. Ajustar o pré-processador SOMENTE no treino (X_train_val)
    #    e transformar ambos
    X_train_val_t = preprocessor_obj.fit_transform(X_train_val)
    X_val_t = preprocessor_obj.transform(X_val)

    # 5. Treinar o modelo com early stopping e pruning
    model.fit(
        X_train_val_t, y_train_val,
        eval_set=[(X_val_t, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            # Corrigido para 'auc' (o nome da métrica)
            optuna.integration.LightGBMPruningCallback(trial, 'auc')
        ],
        categorical_feature=categorical_feature_indices # Passando os índices!
    )

    # 6. Retornar a melhor pontuação de AUC
    return model.best_score_['valid_0']['auc']


# ---
# Bloco 5: Execução do Estudo de Otimização
# ---
print("\nIniciando busca de hiperparâmetros com Optuna...")
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

# (n_trials=50 é um bom começo, aumente para 100-200 para melhores resultados)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBusca finalizada!")
print(f"Melhor valor de AUC (na validação interna): {study.best_value:.4f}")
print(f"Melhores hiperparâmetros encontrados: {study.best_params}")


# ---
# Bloco 6: Treinamento do Modelo FINAL e Montagem do Pipeline
# ---
# Esta é a abordagem correta para evitar o erro '...do not match'
print("\nTreinando o modelo final com os melhores parâmetros...")

# 1. Pegar os melhores parâmetros
best_params = study.best_params

# 2. Instanciar o modelo final (Sintaxe corrigida: usando '=')
final_model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_jobs=-1,
    random_state=42,
    **best_params,      # Adiciona os parâmetros ótimos (ex: max_depth)
    n_estimators=2000   # Número alto, usaremos early stopping
)

# 3. Ajustar o pré-processador UMA VEZ no X_train
print("Ajustando o pré-processador final...")
# (O 'preprocessor' aqui é o objeto original do Bloco 2)
preprocessor.fit(X_train)

# Captura os nomes das features na ordem correta do preprocessor
feature_names_out = preprocessor.get_feature_names_out()

# 4. Transformar ambos os datasets (Treino e Teste)
print("Transformando dados de treino e teste...")
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

# 5. Treinar o modelo FINAL diretamente nos dados transformados
print("Iniciando treinamento final com early stopping...")
final_model.fit(
    X_train_t, y_train,
    eval_set=[(X_test_t, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=500)
    ],
    categorical_feature=categorical_feature_indices, # Crucial!
    feature_name=list(feature_names_out)
)

# 6. Criar o PIPELINE FINAL para deploy
#    Juntamos o pré-processador (JÁ AJUSTADO) e o modelo (JÁ TREINADO)
print("Montando pipeline final para deploy...")
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # pré-processador 'fitado' no X_train
    ('model', final_model)         # modelo 'fitado' no X_train_t
])

# 7. SALVAR O PIPELINE FINAL (O objeto para o Docker)
# (Usando o nome de arquivo que você especificou)
caminho_pipeline = 'modelo_lgbm.joblib'
print(f"Salvando o pipeline completo em {caminho_pipeline}...")
joblib.dump(final_pipeline, caminho_pipeline)


# ---
# Bloco 7: Avaliação do Pipeline Final
# ---
print("\nIniciando avaliação do pipeline FINAL no conjunto de Teste...")
# Carregamos o pipeline salvo para garantir que funciona
loaded_pipeline = joblib.load(caminho_pipeline)

# O pipeline carregado espera dados "crus" (raw)
y_pred_proba = loaded_pipeline.predict_proba(X_test)[:, 1]
y_pred_class = loaded_pipeline.predict(X_test)

auc_score = roc_auc_score(y_test, y_pred_proba)
acc_score = accuracy_score(y_test, y_pred_class)

print(f"\n--- Resultados da Avaliação FINAL ---")
print(f"AUC-ROC no Teste: {auc_score:.4f}")
print(f"Acurácia no Teste: {acc_score:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_class))

# ---
# Bloco 8: Importância das Features (Bônus)
# ---
print("\nGerando gráfico de importância das features...")
try:
    lgb.plot_importance(final_model, max_num_features=20, figsize=(10, 8))
    plt.title("Importância das Features (Modelo Final)")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Gráfico de importância salvo como 'feature_importance.png'")
except Exception as e:
    print(f"Não foi possível gerar gráfico: {e}")

print("\n--- Processo Concluído ---")

