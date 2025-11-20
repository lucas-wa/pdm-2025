# Use uma imagem base Python
FROM python:3.11-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos de dependências
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código da aplicação
COPY . .

# Expor a porta em que a API vai rodar
EXPOSE 8000

# Comando para inicializar o servidor
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]