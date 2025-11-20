import requests, tqdm, os
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
import aiohttp
import nest_asyncio
from google.cloud import storage

get_headers = { "User-Agent": "Mozilla/5.0" }

storage_client = storage.Client()
bucket_name = os.getenv("BUCKET_NAME", "pdm-2025-knowledge-base")
bucket = storage_client.bucket(bucket_name)

MAX_DOWNLOADS = int(os.getenv("MAX_DOWNLOADS", "10")) 

get_headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

limpar_nome_arquivo = lambda nome_bruto: "".join(
    [char for char in nome_bruto if (char.isalnum() or char in ['-', '_'])]
)

def get_all_links_from_knowledge_base():
    ref = "https://atendimento.tron.com.br/kb/article/140004/bem-vindo"

    response = requests.get(ref, headers=get_headers)

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {ref}")

    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("li", class_="Article")

    main_cat, sub_cat, title, link = [], [], [], []

    for article in articles:
        main_cat.append(
            article.find_parent()
            .find_parent()
            .find_parent()
            .find_parent()
            .find("a")
            .text
        )
        sub_cat.append(article.find_parent().find_parent().find("a").text)
        title.append(article.text)
        link.append(article.find("a")["href"])

    return pd.DataFrame(
        {"main_cat": main_cat, "sub_cat": sub_cat, "title": title, "link": link}
    )


async def fetch_and_upload_html(url, blob_name):
    """Baixa HTML e envia para o bucket, se ainda não existir"""
    blob = bucket.blob(blob_name)

    if blob.exists(): 
        print(f"[SKIP] Já existe no bucket: {blob_name}")
        return

    async with aiohttp.ClientSession() as session:
        erros = 0
        while erros < 5:
            async with session.get(url, headers=get_headers) as response:
                if response.status == 200:
                    html = await response.text()
                    blob.upload_from_string(html, content_type="text/html")
                    print(f"[OK] Upload concluído: {blob_name}")
                    return
                else:
                    erros += 1
                    print(f"Error: {response.status} - {url}")


async def get_html_pages(df, bucket_folder="bronze"):
    tasks = []
    total_downloads = 0

    for title, link in zip(df["title"], df["link"]):
        file_name = "".join([c for c in link if c.isalnum() or c in ['-', '_']])
        blob_name = f"bronze/{file_name}/{file_name}.html"

        if MAX_DOWNLOADS > 0 and total_downloads >= MAX_DOWNLOADS:
            print(f"[INFO] Limite de {MAX_DOWNLOADS} downloads atingido.")
            break

        tasks.append(fetch_and_upload_html(link, blob_name))
        total_downloads += 1

        if len(tasks) >= 50:
            await asyncio.gather(*tasks)
            tasks = []

    if tasks:
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    print("[bronze knowledge_base] - Executando o arquivo")
    knowledge_df = get_all_links_from_knowledge_base()
    print("[bronze knowledge_base] - Total de documentos encontrados: ", len(knowledge_df))
    nest_asyncio.apply()
    asyncio.run(get_html_pages(df=knowledge_df, bucket_folder="bronze"))
