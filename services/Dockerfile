FROM python:3.10.14

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers

COPY requirements.txt .

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

#Bronze

# mu

CMD ["sh", "-c", "python", "./bronze/tickets/tickets.py", "&&", "python", "./silver/silver_process_tickets/input_output_batch.py"]

