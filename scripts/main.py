from pyspark.sql.types import StructType, StructField, StringType
import os
from google.cloud import storage, bigquery
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, current_timestamp
from pyspark.sql.types import StringType, ArrayType, FloatType
import requests
import json

GEMINI_MODEL = "gemini-2.5-flash"  
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def get_gemini_embedding(text: str) -> list:
    """
    Usa a API Gemini para gerar embeddings do texto.
    Retorna lista de floats.
    """
    body = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [
                {"text": text}
            ]
        }
    }
    
    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}
    
    try:
        resp = requests.post(GEMINI_EMBEDDING_URL, params=params, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        # Extrair embedding da resposta
        embedding = data.get("embedding", {}).get("values", [])
        
        if not embedding:
            print(f"Embedding vazio retornado para texto: {text[:100]}...")
            return [0.0] * 768  # fallback com dimensão padrão
            
        return embedding
        
    except requests.exceptions.RequestException as e:
        print(f"Erro ao obter embedding do Gemini: {e}")
        return [0.0] * 768
    except Exception as e:
        print(f"Erro inesperado ao obter embedding: {e}")
        return [0.0] * 768


def classify_text(text: str) -> str:
    """
    Usa Gemini para classificar o texto entre 'legislacao' ou 'sistema'.
    Retorna o rótulo como string.
    """
    prompt = (
        "Você é um classificador. Classifique o texto abaixo como 'legislacao' ou 'sistema'. "
        "Responda somente com o rótulo, sem explicações adicionais.\n\n"
        f"Texto: {text[:1000]}"  # Limitar tamanho para evitar tokens excessivos
    )

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(GEMINI_API_URL, params=params, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        # Extrair texto da resposta
        candidate = data.get("candidates", [{}])[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        if parts:
            label = parts[0].get("text", "").strip().lower()
        else:
            label = ""
        
        # Normalizar resposta
        if "legisla" in label:
            return "legislacao"
        elif "sistema" in label:
            return "sistema"
        else:
            print(f"Resposta inesperada do Gemini: {label}")
            return "sistema"
            
    except requests.exceptions.RequestException as e:
        print(f"Erro na chamada Gemini para classificação: {e}")
        return "sistema"
    except Exception as e:
        print(f"Erro inesperado ao classificar via Gemini: {e}")
        return "sistema"


# --- Pipeline com Spark ---
def main():
    spark = SparkSession.builder \
      .appName("EmbeddingsPipeline") \
      .config("spark.rpc.message.maxSize", "2047") \
      .getOrCreate()

    bucket = os.getenv("BUCKET_NAME", "pdm-2025-knowledge-base")
    
    # Leitura do CSV de controle (se existir)
    path_control = f"gs://{bucket}/silver/processed_control"
    try:
        df_ctrl = spark.read.option("header", "true").csv(path_control)
        processed = [r["file_name"] for r in df_ctrl.select("file_name").collect()]
        print(f"Arquivos já processados: {len(processed)}")
    except Exception as e:
        print(f"Controle não encontrado ou erro: {e}")
        processed = []

    # Listagem dos arquivos .txt no bucket
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    prefix = "knowledge_base"
    print(f"Listando arquivos no bucket gs://{bucket}/{prefix}...")
    
    blobs = client.list_blobs(bucket, prefix=prefix)
    files = []
    for blob in blobs:
        name = blob.name
        if name.lower().endswith(".txt"):
            fname = os.path.basename(name)
            files.append((fname, name))
    
    print(f"Total de arquivos .txt encontrados: {len(files)}")
    
    if not files:
        print("AVISO: Nenhum arquivo .txt encontrado no bucket!")
        spark.stop()
        return
    
    # Criar DataFrame com os dados
    schema = StructType([
        StructField("file_name", StringType(), True),
        StructField("blob_path", StringType(), True)
    ])
    
    df_files = spark.createDataFrame(files, schema)
    print(f"DataFrame inicial criado: {df_files.count()} linhas")
    
    # Mostrar alguns exemplos
    print("Exemplos de arquivos:")
    df_files.show(5, truncate=False)
    
    # Filtrar arquivos já processados
    if processed:
        df_to_process = df_files.filter(~col("file_name").isin(processed))
    else:
        df_to_process = df_files
    
    count_to_process = df_to_process.count()
    print(f"Arquivos a processar: {count_to_process}")
    
    # Se não há nada para processar, encerra
    if count_to_process == 0:
        print("Nenhum arquivo novo para processar")
        spark.stop()
        return

    # UDF para ler texto do GCS
    def read_text(blob_path: str) -> str:
        from google.cloud import storage as gcs_storage
        try:
            client2 = gcs_storage.Client()
            bucket2 = client2.bucket(bucket)
            blob2 = bucket2.blob(blob_path)
            text = blob2.download_as_text(encoding="utf-8")
            print(f"Lido com sucesso: {blob_path} ({len(text)} chars)")
            return text
        except Exception as e:
            print(f"Erro ao ler {blob_path}: {e}")
            return ""

    read_udf = udf(read_text, StringType())
    df_txt = df_to_process.withColumn("text", read_udf("blob_path"))
    
    print("Textos carregados, processando embeddings e classificações...")

    # UDF para processar embedding e classificação
    def compute_embedding_and_class(text: str) -> str:
        if not text or len(text.strip()) == 0:
            return json.dumps({
                "classification": "sistema",
                "embedding": [0.0] * 768
            })
        
        print(f"Processando texto de {len(text)} caracteres...")
        
        # Obter embedding via Gemini
        emb = get_gemini_embedding(text)
        
        # Classificar via Gemini
        cls = classify_text(text)
        
        result = {
            "classification": cls,
            "embedding": emb
        }
        
        print(f"Resultado: classificação={cls}, embedding_dim={len(emb)}")
        
        return json.dumps(result)

    udf_emb_class = udf(compute_embedding_and_class, StringType())
    df_res = df_txt.withColumn("out", udf_emb_class("text"))

    # Separar colunas
    def extract_cls(s: str) -> str:
        try:
            d = json.loads(s)
            return d.get("classification", "sistema")
        except:
            return "sistema"
    
    def extract_emb_list(s: str) -> list:
        try:
            d = json.loads(s)
            return d.get("embedding", [])
        except:
            return []

    udf_cls = udf(extract_cls, StringType())
    udf_emb = udf(extract_emb_list, ArrayType(FloatType()))

    df_final = df_res \
        .withColumn("classification", udf_cls("out")) \
        .withColumn("embedding", udf_emb("out")) \
        .select("file_name", "blob_path", "text", "classification", "embedding")
    
    final_count = df_final.count()
    print(f"DataFrame final: {final_count} linhas")
    
    # Mostrar exemplos do resultado
    print("Exemplos de resultados:")
    df_final.select("file_name", "classification").show(5, truncate=False)
    
    # Gravar controle (file_name + classification + timestamp)
    df_ctrl_new = df_final.select(
        "file_name", 
        "classification",
        current_timestamp().alias("processed_at")
    )
    
    print(f"Salvando controle com {df_ctrl_new.count()} registros...")
    df_ctrl_new.write.mode("append").option("header", "true").csv(path_control)
    print("Controle salvo!")

    # Preparar dados para BigQuery
    # Converter embedding para string JSON
    from pyspark.sql.functions import to_json
    
    df_to_bq = df_final.select(
        "file_name",
        "blob_path", 
        "classification",
        to_json(col("embedding")).alias("embedding_json"),
        col("text").substr(1, 500).alias("text_preview")  # Preview dos primeiros 500 chars
    )
    
    print(f"Preparando {df_to_bq.count()} registros para BigQuery...")
    
    # Mostrar schema
    print("Schema para BigQuery:")
    df_to_bq.printSchema()
    
    # Gravar no BigQuery
    table_id = "valid-task-471913-e7.aula_pdm.embeddings_table"
    
    try:
        df_to_bq.write \
            .format("bigquery") \
            .option("table", table_id) \
            .option("temporaryGcsBucket", bucket) \
            .mode("append") \
            .save()
        
        print(f"✓ Dados gravados no BigQuery com sucesso!")
    except Exception as e:
        print(f"✗ Erro ao gravar no BigQuery: {e}")

    spark.stop()
    print("Pipeline finalizado!")


if __name__ == "__main__":
    main()