import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go  # Importado para o 'add_shape'
import pandas as pd
import numpy as np  # Importado para a simulação
import requests
import os
import base64  # Necessário para o upload
import io  # Necessário para o upload

# --- 1. Configuração ---

# O URL da sua API Docker
API_URL = "http://localhost:8000/predict"

# O arquivo de dados local (usado para os histogramas de base)
NOME_ARQUIVO_PARQUET = 'influenza_ML_2025-10-30_16-28-12.parquet'

# As 6 variáveis que queremos destacar na tabela final
VAR_IMPORTANTES_QUANT = [
    'IDADE',
    'DIAS_INTERNACAO',
    'DIAS_SINT_INI_RX_RESP',
    'DIAS_INTERNA_RX_RESP',
    'DIAS_UL_INIC_ANTIVIRAL'
]
VAR_IMPORTANTE_CAT = ['RAIOX_RES']
VAR_IMPORTANTES = VAR_IMPORTANTES_QUANT + VAR_IMPORTANTE_CAT

# As colunas de features que o seu modelo espera (com base no treino_pipeline.py)
# A API *precisa* de todas estas colunas para o pré-processamento
COLUNAS_FEATURES = [
    'CS_SEXO', 'CS_GESTANT', 'CS_RACA', 'CS_ESCOL_N', 'ID_PAIS', 'CS_ZONA',
    'NOSOCOMIAL', 'AVE_SUINO', 'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP',
    'SATURACAO', 'DIARREIA', 'VOMITO', 'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI',
    'SIND_DOWN', 'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI',
    'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'VACINA', 'MAE_VAC', 'M_AMAMENTA',
    'ANTIVIRAL', 'TP_ANTIVIR', 'HOSPITAL', 'RAIOX_RES', 'PCR_RESUL',
    'POS_PCRFLU', 'TP_FLU_PCR', 'PCR_FLUASU', 'PCR_FLUBLI', 'POS_PCROUT',
    'PCR_VSR', 'PCR_PARA1', 'PCR_PARA2', 'PCR_PARA3', 'PCR_PARA4', 'PCR_ADENO',
    'PCR_METAP', 'PCR_BOCA', 'PCR_RINO', 'PCR_OUTRO', 'DOR_ABD', 'FADIGA',
    'PERD_OLFT', 'PERD_PALA', 'TOMO_RES', 'TP_TES_AN', 'RES_AN', 'POS_AN_FLU',
    'TP_FLU_AN', 'POS_AN_OUT', 'AN_SARS2', 'AN_VSR', 'AN_PARA1', 'AN_PARA2',
    'AN_PARA3', 'AN_ADENO', 'AN_OUTRO', 'POV_CT', 'SURTO_SG', 'IDADE',
    'DIAS_UL_VAC', 'DIAS_UL_VAC_MAE', 'DIAS_UL_VAC_DOSEUNI', 'DIAS_UL_VAC_1_DOSE',
    'DIAS_UL_VAC_2_DOSE', 'DIAS_UL_INIC_ANTIVIRAL', 'DIAS_INTERNACAO',
    'DIAS_INTERNA_RX_RESP', 'DIAS_SINT_INI_RX_RESP', 'DIAS_INTERNA_TOMO',
    'DIAS_SINT_INI_TOMO'
]
COLUNA_ALVO_BASE = 'UTI'  # Coluna alvo usada apenas para o arquivo de base

# --- 2. Carregamento Global dos Dados de Base ---
# Carrega os dados do parquet para os histogramas de base
try:
    print(f"A carregar dados de base de {NOME_ARQUIVO_PARQUET}...")
    df_base = pd.read_parquet(NOME_ARQUIVO_PARQUET)

    # Limpeza de dados (mimicando o seu treino_pipeline.py)
    cols_to_drop = ['DT_SIN_PRI', 'OBES_IMC', 'FEBRE', 'TABAG']
    df_base.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df_base.drop(df_base.loc[df_base[COLUNA_ALVO_BASE] == 9].index, axis=0, inplace=True)
    df_base.drop(df_base.loc[df_base[COLUNA_ALVO_BASE].isnull()].index, axis=0, inplace=True)
    df_base[COLUNA_ALVO_BASE] = df_base[COLUNA_ALVO_BASE].map({1.0: 1, 2.0: 0})

    # Filtrar apenas para os casos reais de UTI (para a população de referência)
    df_total_uti_casos = df_base[df_base[COLUNA_ALVO_BASE] == 1].copy()

    # Converter idade dos dados de base para ANOS
    df_total_uti_casos['IDADE'] = df_total_uti_casos['IDADE'] / 365.25

    print(f"Dados de base (UTI=1) carregados: {len(df_total_uti_casos)} registos.")

except Exception as e:
    print(f"Erro Crítico: Não foi possível carregar os dados de base {NOME_ARQUIVO_PARQUET}.")
    print(f"Erro: {e}")
    df_total_uti_casos = pd.DataFrame(columns=VAR_IMPORTANTES_QUANT)  # Cria um dataframe vazio

# --- 3. Inicialização da App Dash ---

# *** ALTERAÇÃO: Adicionado suppress_callback_exceptions=True para corrigir o erro ***
app = dash.Dash(
    __name__,
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
    suppress_callback_exceptions=True
)
app.title = "Dashboard Predição de Casos"


# --- 4. Funções Auxiliares ---

def parse_csv_contents(contents, filename):
    """Lê o conteúdo de um arquivo CSV carregado e retorna um DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, html.Div(['Por favor, carregue um arquivo .csv.'])
    except Exception as e:
        print(e)
        return None, html.Div(['Ocorreu um erro ao processar este arquivo.'])

    return df, None


def create_base_histograms():
    """Cria os 5 histogramas de base sem linhas verticais."""
    fig_idade = px.histogram(df_total_uti_casos, x="IDADE", title="Distribuição de Idade (Anos)", nbins=50)
    fig_internacao = px.histogram(df_total_uti_casos, x="DIAS_INTERNACAO", title="Dias de Internação")
    fig_sint_rx = px.histogram(df_total_uti_casos, x="DIAS_SINT_INI_RX_RESP", title="Dias (Sintoma -> Raio-X)")
    fig_int_rx = px.histogram(df_total_uti_casos, x="DIAS_INTERNA_RX_RESP", title="Dias (Internação -> Raio-X)")
    fig_antiviral = px.histogram(df_total_uti_casos, x="DIAS_UL_INIC_ANTIVIRAL", title="Dias para Iniciar Antiviral")
    return [fig_idade, fig_internacao, fig_sint_rx, fig_int_rx, fig_antiviral]


# --- 5. Layout da Aplicação ---

# *** ALTERAÇÃO: Gera os gráficos de base *antes* do layout ***
figs_base = create_base_histograms()

app.layout = html.Div(style={'padding': '20px'}, children=[

    # Armazém oculto para guardar os dados da predição
    dcc.Store(id='predicoes-store'),

    html.H1(
        children='Dashboard de Predição e Análise de Casos de UTI',
        style={'textAlign': 'center', 'color': '#333'}
    ),
    html.Hr(),

    # Secção 1: Upload
    html.H2("1. Carregar Arquivo CSV de Pacientes"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arraste e Solte ou ',
            html.A('Selecione um Arquivo CSV')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center',
            'marginBottom': '20px'
        },
        multiple=False
    ),

    html.Hr(),

    # Secção 2: Histogramas
    html.H2("2. Distribuição da População Total (UTI=1) vs. Caso Selecionado"),
    html.P(
        "Os histogramas mostram a população total (casos UTI=1). Clique numa linha da tabela abaixo para ver o caso correspondente (linha vertical colorida)."),

    # --- LINHA 1: 3 Histogramas ---
    html.Div(className='row', children=[
        # *** ALTERAÇÃO: Gráficos pré-carregados com 'figure=...' ***
        html.Div(className='four columns', children=[
            dcc.Graph(id='hist-idade', figure=figs_base[0])
        ]),
        html.Div(className='four columns', children=[
            dcc.Graph(id='hist-internacao', figure=figs_base[1])
        ]),
        html.Div(className='four columns', children=[
            dcc.Graph(id='hist-sint-rx', figure=figs_base[2])
        ]),
    ]),

    # --- LINHA 2: 2 Histogramas ---
    html.Div(className='row', style={'marginTop': '20px'}, children=[
        html.Div(className='six columns', children=[
            dcc.Graph(id='hist-int-rx', figure=figs_base[3])
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(id='hist-antiviral', figure=figs_base[4])
        ]),
    ]),

    html.Hr(style={'marginTop': '30px', 'marginBottom': '30px'}),

    # Secção 3: Saída da Tabela
    html.H2("3. Resultados da Predição (Arquivo Carregado)"),
    html.Div(
        id='output-tabela-container',
        children=[
            html.P("Por favor, carregue um arquivo CSV para ver os resultados da predição.")
        ],
        style={
            'padding': '10px',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'background': '#f9f9f9'
        }
    ),

    html.Hr(style={'marginTop': '30px', 'marginBottom': '30px'}),

    # Secção 4: Simulação
    html.H2("4. Simulação de Casos Esperados"),
    html.P("Histograma do número total de casos de UTI esperados (do lote carregado), baseado em 10.000 simulações."),
    # *** ALTERAÇÃO: Gráfico de simulação começa vazio ***
    dcc.Graph(id='hist-simulacao', figure=go.Figure())
])


# --- 6. Callbacks da Aplicação ---

# Callback 1: Lida com o Upload do Arquivo
# Atualiza a Tabela, o Armazém de Dados (Store) e o Histograma de Simulação
@app.callback(
    Output('output-tabela-container', 'children'),
    Output('predicoes-store', 'data'),
    Output('hist-simulacao', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True  # Não executa quando a app inicia
)
def update_on_upload(contents, filename):
    # Estado inicial (sem arquivo carregado)
    if contents is None:
        return html.P("Por favor, carregue um arquivo CSV."), None, go.Figure()

    # --- Se um arquivo foi carregado, processa ---
    df, error_message = parse_csv_contents(contents, filename)
    if error_message:
        return error_message, None, go.Figure()

    df_casos_novos = df.copy()

    if df_casos_novos.empty:
        return html.P(f"Nenhum caso válido encontrado no arquivo '{filename}'."), None, go.Figure()

    print(f"Encontrados {len(df_casos_novos)} casos no arquivo carregado para predição.")
    # Adiciona um ID de paciente genérico, assumindo que o CSV não tem um
    if 'ID_Paciente' not in df_casos_novos.columns:
        df_casos_novos['ID_Paciente'] = df_casos_novos.index

    # 4. Preparar dados e chamar a API
    try:
        X_casos = df_casos_novos[COLUNAS_FEATURES]
    except KeyError as e:
        error_msg = f"Erro: O arquivo CSV não contém todas as colunas de features necessárias. Coluna em falta: {e}"
        return html.P(error_msg), None, go.Figure()

    print(f"A contactar a API em {API_URL}...")
    try:
        json_payload = X_casos.to_json(orient='records')
        response = requests.post(
            API_URL,
            data=json_payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        results = response.json()
        probabilities = results.get('predictions_prob_class_1')
        print("Predições recebidas com sucesso.")

    except requests.exceptions.ConnectionError:
        error_msg = f"ERRO DE CONEXÃO: Não foi possível ligar à API em {API_URL}. O container Docker está a ser executado?"
        return html.P(error_msg), None, go.Figure()
    except Exception as e:
        error_msg = f"Erro ao chamar a API: {str(e)}"
        return html.P(error_msg), None, go.Figure()

    # 5. Montar a tabela de display final
    df_display = df_casos_novos[['ID_Paciente'] + VAR_IMPORTANTES].copy()
    df_display['Prob_Raw'] = probabilities
    df_display['Probabilidade_UTI'] = df_display['Prob_Raw'].apply(lambda x: f"{x * 100:.1f}%")

    # Converter IDADE para ANOS para a tabela
    df_display['IDADE'] = (df_display['IDADE'] / 365.25).round(1)

    df_display = df_display[['ID_Paciente', 'Probabilidade_UTI', 'Prob_Raw'] + VAR_IMPORTANTES]
    df_display.sort_values(by='Prob_Raw', ascending=False, inplace=True)

    colunas_tabela = [{'name': i, 'id': i} for i in df_display.columns if i != 'Prob_Raw']

    # 6. Gerar a Tabela
    output_tabela = html.Div([
        html.H3(f"Predições para {len(df_casos_novos)} Casos - Arquivo: {filename}"),
        html.P("Clique numa linha para destacar o caso nos gráficos acima."),

        dash_table.DataTable(
            id='tabela-predicoes',
            data=df_display.to_dict('records'),
            columns=colunas_tabela,
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Prob_Raw} >= 0.5', 'column_id': 'Probabilidade_UTI'},
                    'backgroundColor': '#FF4136',  # Vermelho
                    'color': 'white'
                },
                {
                    'if': {'filter_query': '{Prob_Raw} < 0.5', 'column_id': 'Probabilidade_UTI'},
                    'backgroundColor': '#2ECC40',  # Verde
                    'color': 'white'
                }
            ],
            page_size=15,
            sort_action='native',
            filter_action='native',
            row_selectable="single",  # Permite selecionar uma linha
            selected_rows=[],  # Começa sem seleção
        )
    ])

    # 7. Gerar o Histograma de Simulação
    print("A executar simulação de Monte Carlo...")
    probabilities_array = df_display['Prob_Raw'].values
    n_pacientes = len(probabilities_array)
    n_ensaios = 10000

    simulacoes = np.random.binomial(1, probabilities_array, size=(n_ensaios, n_pacientes))
    total_por_ensaio = np.sum(simulacoes, axis=1)

    # --- ALTERAÇÕES NA SIMULAÇÃO ---

    # Calcular IC de 95%
    ci_lower = np.percentile(total_por_ensaio, 2.5)
    ci_upper = np.percentile(total_por_ensaio, 97.5)
    media_esperada = np.mean(total_por_ensaio)

    titulo_sim = (
        f"Distribuição de Casos Positivos Esperados (de {n_pacientes} pacientes)"
        f"<br>Média: {media_esperada:.2f} | IC 95%: [{ci_lower:.0f} - {ci_upper:.0f}]"
    )

    # Calcular o número de 'bins' (barras) de forma dinâmica
    min_bins = 20
    # Garante que max_bins_dinamico seja um inteiro e pelo menos 1
    max_bins_dinamico = max(1, 1 + int(total_por_ensaio.max()) - int(total_por_ensaio.min()))
    nbins = max(min_bins, max_bins_dinamico)

    fig_sim = px.histogram(
        x=total_por_ensaio,
        nbins=nbins,
        title=titulo_sim,
        histnorm='probability density'  # Mostra proporção
    )

    # Adiciona linhas para o IC de 95%
    fig_sim.add_shape(
        type="line", xref="x", yref="paper",
        x0=ci_lower, y0=0, x1=ci_lower, y1=1,
        line=dict(color="orange", width=2, dash="dash")
    )
    fig_sim.add_shape(
        type="line", xref="x", yref="paper",
        x0=ci_upper, y0=0, x1=ci_upper, y1=1,
        line=dict(color="orange", width=2, dash="dash")
    )

    fig_sim.update_layout(
        xaxis_title="Número Total de Casos de UTI por Simulação",
        yaxis_title="Proporção / Densidade de Probabilidade"
    )
    # --- FIM DAS ALTERAÇÕES NA SIMULAÇÃO ---

    # 8. Retornar tudo
    return output_tabela, df_display.to_dict('records'), fig_sim


# Callback 2: Lida com a atualização dos gráficos (baseado na Tabela ou no Upload)
# Atualiza os 5 histogramas
@app.callback(
    Output('hist-idade', 'figure'),
    Output('hist-internacao', 'figure'),
    Output('hist-sint-rx', 'figure'),
    Output('hist-int-rx', 'figure'),
    Output('hist-antiviral', 'figure'),
    Input('predicoes-store', 'data'),  # Dispara quando o upload termina
    Input('tabela-predicoes', 'selected_rows'),  # Dispara quando uma linha é clicada
    State('tabela-predicoes', 'data')  # Obtém os dados *atuais* da tabela (após ordenação/filtro)
)
def update_graphs_on_selection(predicoes_data, selected_rows, table_data):
    # 1. Gerar sempre os gráficos de base
    figs = create_base_histograms()
    fig_idade, fig_internacao, fig_sint_rx, fig_int_rx, fig_antiviral = figs

    # 2. Verificar se há dados e uma linha selecionada
    # Se 'selected_rows' for vazio (ex: [ ]), não faz nada
    if not selected_rows or not table_data:
        return figs

    # 3. Se uma linha for selecionada, obter os seus dados
    try:
        selected_row_index = selected_rows[0]  # Pega o índice da primeira (e única) linha

        # Usa 'table_data' (State) para obter os dados da linha que o utilizador VÊ
        # (Isto respeita a ordenação/filtragem da tabela)
        selected_patient = table_data[selected_row_index]

        line_color = "red" if selected_patient['Prob_Raw'] >= 0.5 else "green"

        # 4. Adicionar as linhas verticais (usando add_shape para corrigir o bug do zoom)
        print(f"A destacar Paciente ID: {selected_patient['ID_Paciente']}")

        # Lista dos gráficos e das variáveis correspondentes
        graphs_vars = [
            (fig_idade, 'IDADE'),
            (fig_internacao, 'DIAS_INTERNACAO'),
            (fig_sint_rx, 'DIAS_SINT_INI_RX_RESP'),
            (fig_int_rx, 'DIAS_INTERNA_RX_RESP'),
            (fig_antiviral, 'DIAS_UL_INIC_ANTIVIRAL')
        ]

        for fig, var_name in graphs_vars:
            if var_name in selected_patient and pd.notna(selected_patient[var_name]):
                x_val = selected_patient[var_name]
                fig.add_shape(
                    type="line",
                    xref="x", yref="paper",  # Prende ao eixo-x (dados) e ao eixo-y (papel)
                    x0=x_val, y0=0,  # x=valor, y=fundo
                    x1=x_val, y1=1,  # x=valor, y=topo
                    line=dict(color=line_color, width=2, dash="dash")
                )

    except Exception as e:
        print(f"Erro ao tentar destacar linha: {e}")
        # Retorna os gráficos base em caso de erro
        return figs

    return figs


# --- 7. Executar a Aplicação ---
if __name__ == '__main__':
    print(f"\nDashboard a ser executado em http://127.0.0.1:8050/")
    print(f"A API Docker deve estar a ser executada em {API_URL}")
    app.run(debug=True, port=8050)

