import pandas as pd
from FlagEmbedding import FlagModel
import time
import os
import json
import datetime

PASTA_TICKETS_PROCESSADOS = '/silverII/'

# Carregar todos os arquivos JSON da pasta
tickets_files = [f for f in os.listdir(PASTA_TICKETS_PROCESSADOS) if f.endswith('.json')]

# Inicializar uma lista para armazenar os tickets
tickets_data = []

for ticket_file in tickets_files:
    file_path = os.path.join(PASTA_TICKETS_PROCESSADOS, ticket_file)
    with open(file_path, 'r', encoding='utf-8') as file:
        ticket = json.load(file)
        tickets_data.append(ticket)

# Criar o DataFrame com os tickets carregados
df_tickets = pd.DataFrame(tickets_data)

def juntar_texto(row):
    res = json.loads(row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    print(res)

    campos = [
        res['pergunta_principal'],
        res['analise_pergunta'],
        res['orientacao_fornecida'],
        res['status_resolucao'],
        res['roteiro_resolucao']
    ]
    if any(not campo for campo in campos):
        return None
    return (
        f"Para resolver dúvidas que se parecam com essa pergunta: {res['pergunta_principal']}\n"
        f"Você deve analisar que: {res['analise_pergunta']}\n"
        f"E nossos atendentes humanos forneceram a seguinte orientação: {res['orientacao_fornecida']}\n"
        f"Para resolver ela nós temos esse possível roteiro de Resoulção: {res['roteiro_resolucao']}"
        f"E como status da resolução temos: {res['status_resolucao']}"
    )

df_tickets["texto"] = df_tickets.apply(juntar_texto, axis=1)
print(df_tickets[["texto"]].head())

print("iniciando processo de criação de embeddings...")

print("carregando modelo nosso...")
start_time = time.time()
# Definir o diretório de cache
cache_dir = "/cache/"
os.makedirs(cache_dir, exist_ok=True)

# Caminho do modelo no cache
modelo_cache_path = os.path.join(cache_dir, "flag_model")

# Verificar se o modelo já está na pasta de cache

start_time = time.time()
model = FlagModel(
    model_name_or_path="davidoneil/bge-m3-ft-corpus-pt",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True,
    cache_dir=modelo_cache_path
)
print(f"Modelo carregado e salvo em {modelo_cache_path} em {time.time() - start_time:.2f} segundos")


print("preparando textos para embeddings...")
textos = df_tickets['texto'].tolist()
print(f"total de textos: {len(textos)}")
print(f"tamanho médio dos textos: {sum(len(texto) for texto in textos) / len(textos):.0f} caracteres")

print("\primeiro 3 textos:")
for i, texto in enumerate(textos[:3]):
    print(f"  {i+1}. {texto[:100]}...")

print("\ncreate embeddings...")
start_time = time.time()
response = model.encode_corpus(textos)
print("textos", len(textos))
print("response", len(response))
embeddings = response
print(f"embeddigns criados em {time.time() - start_time:.2f} segundos")
print(f"format dos embeddings: {embeddings.shape}")

print("add embeddings ao DataFrame...")
df_tickets['embeddings'] = embeddings.tolist()
print(f" Colunas do DataFrame: {list(df_tickets.columns)}")
print(f" Forma final do DataFrame: {df_tickets.shape}")


timestamp = int(datetime.datetime.now().timestamp())
print(" salalvando dados...")
df_tickets.to_pickle(f'/gold/{timestamp}_tickets_embeddings.pkl')
print(f" daddos salvos em '/gold/{timestamp}_tickets_embeddings.pkl'")

print("\nverificando:")
print(f"registros salvos: {len(df_tickets)}")
print(f"primeira embedding shape: {len(df_tickets.iloc[0]['embeddings'])}")
print("processo concluído")