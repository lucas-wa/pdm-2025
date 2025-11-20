FROM python:3.10.14

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . .

CMD ["python", "main.py"]

