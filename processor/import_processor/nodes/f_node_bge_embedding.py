import json
from pathlib import Path
from typing import List, Dict

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        chunks = self._step_1_validate_path(state)
        output_data = self._step_generate_embedding(chunks)

        # 先跑通看结果：暂不落盘备份
        # self._step_backup_chunks(state, output_data)

        state["chunks"] = output_data
        return state

    def _step_1_validate_path(self, state: ImportGraphState) -> List[Dict]:
        print("node_bge_embedding:步骤1:参数校验")
        chunks = state.get("chunks")

        if not chunks:
            raise ValueError("参数错误:chunks")

        if not isinstance(chunks, list):
            raise StateFieldError(
                field_name="chunks",
                message="chunks数据类型不正确",
                expected_type=list,
            )
        return chunks

    def _step_generate_embedding(self, chunks: List[Dict]) -> List[Dict]:
        print("node_bge_embedding:步骤2:数据向量化")
        output_data = []
        batch_size = 5

        for i in range(0, len(chunks), batch_size):
            five_text = chunks[i:i + batch_size]
            five_ready_xlh_texts = []
            for doc in five_text:
                item_name = doc.get("item_name", "") or ""
                content = doc.get("content", "") or ""
                # 商品名前置，强化检索特征
                five_ready_xlh_texts.append(
                    f"{item_name}\n{content}" if item_name else content
                )

            embedding = generate_embeddings(five_ready_xlh_texts)

            for j, doc in enumerate(five_text):
                item = doc.copy()
                item["dense_vector"] = embedding["dense"][j]
                item["sparse_vector"] = embedding["sparse"][j]
                output_data.append(item)

        return output_data

    def _step_backup_chunks(self, state, chunks):
        """
        备份加入稠密/稀疏向量后的文档切片。

        输出文件：
          {file_title}_new_new_chunks.json
          例如 file_title=B530_new → B530_new_new_new_chunks.json
          例如 file_title=hak180_new → hak180_new_new_new_chunks.json
        """
        print("node_bge_embedding:步骤3:备份向量化后的chunks")
        backup_dir = self._resolve_backup_dir(state)
        file_title = state.get("file_title") or "unknown"
        path = backup_dir / f"{file_title}_new_new_chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        self.logger.info(f"加入稠密稀疏向量后的文档已备份: {path}")
        return str(path)

    @staticmethod
    def _resolve_backup_dir(state) -> Path:
        md_path = state.get("md_path")
        if md_path:
            return Path(md_path).parent
        file_dir = state.get("file_dir")
        if file_dir:
            return Path(file_dir)
        raise StateFieldError(
            field_name="md_path/file_dir",
            message="无法确定备份目录，请提供 md_path 或 file_dir",
            expected_type=str,
        )


if __name__ == "__main__":
    node = NodeBGEEmbedding()
    # 先用 d 节点已有切片跑通（暂不依赖 e 的备份文件）
    with open(r"D:\output\hak180\hak180_new_chunks.json", "r", encoding="utf-8") as f:
        chunks_content = f.read()

    json_state = json.loads(chunks_content)
    init_state = {
        "file_title": "hak180_new",
        "md_path": r"D:\output\hak180\hak180_new.md",
        "file_dir": r"D:\output\hak180",
        "chunks": json_state,
    }

    response = node(init_state)
    chunks = response.get("chunks", [])
    print(f"向量化完成，共 {len(chunks)} 条")
    if chunks:
        c0 = chunks[0]
        dense = c0.get("dense_vector") or []
        sparse = c0.get("sparse_vector") or {}
        print(f"第1条 title: {c0.get('title')}")
        print(f"第1条 item_name: {c0.get('item_name')}")
        print(f"第1条 dense 维度: {len(dense)}")
        print(f"第1条 dense 前5维: {dense[:5]}")
        print(f"第1条 sparse 非零项数: {len(sparse)}")
        # 只打印摘要，避免整段向量刷屏
        summary = []
        for c in chunks:
            summary.append({
                "title": c.get("title"),
                "item_name": c.get("item_name"),
                "dense_dim": len(c.get("dense_vector") or []),
                "sparse_nnz": len(c.get("sparse_vector") or {}),
            })
        print(json.dumps(summary, ensure_ascii=False, indent=2))

