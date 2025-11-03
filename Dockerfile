# 1. Imagem Base
FROM python:3.12-slim

# 2. Instalar a dependência do sistema (libgomp)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 3. NOVO PASSO: Desativar o buffering do Python
#    Isto força os comandos print() a aparecerem imediatamente
ENV PYTHONUNBUFFERED=1

# 4. Definir o diretório de trabalho
WORKDIR /app

# 5. Copiar o arquivo de dependências
COPY requirements.txt .

# 6. Instalar as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copiar o resto do seu código e o modelo
COPY . .

# 8. Expor a porta
EXPOSE 8000

# 9. CMD CORRIGIDO: Comando para rodar a aplicação
#    --log-level=debug: Mostra mais detalhes
#    --access-logfile=- : Envia logs de acesso (GET/POST) para o stdout
#    --error-logfile=- : Envia logs de erro (como o nosso) para o stderr
CMD ["gunicorn", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--log-level=debug", \
     "--access-logfile=-", \
     "--error-logfile=-", \
     "app:app"]
