# Dockerfile

# 1. Imagem Base
# Use a mesma versão do Python do seu ambiente de treino (3.12)
FROM python:3.12-slim

# 2. Definir o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copiar o arquivo de dependências
COPY requirements.txt .

# 4. Instalar as dependências
# --no-cache-dir economiza espaço
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o resto do seu código e o modelo para dentro do container
COPY . .

# 6. Expor a porta que o Gunicorn (servidor de produção) usará
EXPOSE 8000

# 7. Comando para rodar a aplicação
# Usamos Gunicorn, um servidor WSGI robusto para produção
# app:app significa "no arquivo 'app.py', rode a variável 'app' (do Flask)"
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "app:app"]
