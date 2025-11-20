import os
from google.cloud import storage
from bs4 import BeautifulSoup

def process_html_to_text(html_content):
    """Dada uma string HTML, extrai o conteúdo de texto desejado."""
    soup = BeautifulSoup(html_content, "html.parser")
    div = soup.find('article', {'id': 'kb-article'})
    if div is None:
        return ""
    for sub in div.find_all('div', {'class': 'rating-box-form'}):
        sub.decompose()
    return div.get_text()

def convert_html_blobs_to_txt(bucket_name, prefix_html_folder):
    """
    Percorre todos os blobs HTML sob `prefix_html_folder`,
    extrai texto e salva .txt no **mesmo local** (diretório virtual do bucket).
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = client.list_blobs(bucket_name, prefix=prefix_html_folder)

    for blob in blobs:
        name = blob.name  # ex: "bronze/knowledge_base/foo123/foo123.html"
        if not name.lower().endswith(".html"):
            continue

        # Derivar o nome do .txt: substituir .html por .txt
        txt_blob_name = name[:-5] + ".txt"  # remove “.html” e adiciona “.txt”

        txt_blob_name = "silver/" + txt_blob_name.split("/")[-1]

        # Verificar se já existe
        txt_blob = bucket.blob(txt_blob_name)
        if txt_blob.exists():
            print(f"[SKIP] Já existe {txt_blob_name}")
            continue

        # Baixar HTML
        html_content = blob.download_as_text(encoding="utf-8")
        text = process_html_to_text(html_content)

        # Salvar o .txt no bucket
        txt_blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
        print(f"[OK] Criado: {txt_blob_name}")

if __name__ == "__main__":
    BUCKET = os.getenv("BUCKET_NAME", "pdm-2025-knowledge-base")
    PREFIX_HTML = "bronze"  

    convert_html_blobs_to_txt(BUCKET, PREFIX_HTML)
