import dashscope
from dotenv import load_dotenv
from config.reranker_config import reranker_config

load_dotenv()

def rerank_documents(query: str, documents: list[str]) -> list[float]:
    dashscope.api_key = reranker_config.text_rerank_api_key
    response = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct,
    )

    status_code = response.get("status_code")
    if status_code != 200:
        message = response.get("message")
        raise RuntimeError(f"DashScope rerank 调用失败: {message}")

    rerank_scores = [0.0] * len(documents)
    results = (response.get("output") or {}).get("results") or []

    for result in results:
        idx = result.get("index")
        if idx is None:
            continue
        rerank_scores[idx] = result.get("relevance_score")

    print(f"rerank_scores:{rerank_scores}")
    return rerank_scores
