from utils.embedding_utils import generate_embeddings, get_bge_m3_ef

model=get_bge_m3_ef()

embeddings=generate_embeddings("你好")

print(embeddings)