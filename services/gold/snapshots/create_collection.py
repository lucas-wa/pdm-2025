import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from pathlib import Path
import os
import datetime

OUTPUT_PATH = "/goldII/"
input_paths = []
timestamp = int(datetime.datetime.now().timestamp())

gold_path = "/gold/"
input_paths = [
    Path(gold_path, file) for file in os.listdir(gold_path) if file.endswith(".pkl")
]

dataframes = [pd.read_pickle(path) for path in input_paths]
df_concat = pd.concat(dataframes, ignore_index=True)


collection_name = f"tickets_homolog"


print(df_concat.columns)
print(df_concat.head())


qdrant_client = QdrantClient(url="host.docker.internal:6333", prefer_grpc=False)
print("eval", len(df_concat["embeddings"].tolist()[0]))
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=len(df_concat["embeddings"].tolist()[0]), distance="Cosine"
    ),
)

points = []
for idx, row in df_concat.iterrows():
    points.append(
        PointStruct(
            id=idx,
            vector=row["embeddings"], 
            payload={
                "Conteúdo": row["texto"],
            },
        )
    )

qdrant_client.upsert(collection_name=collection_name, points=points)

print(f"Collection '{collection_name}' criada e preenchida com sucesso!")
