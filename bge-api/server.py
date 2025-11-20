# server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from FlagEmbedding import FlagModel
import uvicorn

class EmbeddingRequest(BaseModel):
    texts: list[str]

app = FastAPI()
model = None

@app.on_event("startup")
def startup_event():
    global model
    model = FlagModel(
        model_name_or_path="davidoneil/bge-m3-ft-corpus-pt",
        use_fp16=True,
        cache_dir="/app/cache/flag_model"
    )

@app.post("/embed")
async def embed(req: EmbeddingRequest):
    texts = req.texts
    output = model.encode(texts, batch_size=8, max_length=8192)
    # retornamos apenas dense embeddings como exemplo
    return {"dense_vecs": output['dense_vecs']}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)