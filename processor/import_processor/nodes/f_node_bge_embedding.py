# ==================== 标准库 ====================
# json: 读写备份 JSON
# Path: 拼备份目录/文件路径
# List/Dict: 类型注解
import json
from pathlib import Path
from typing import List, Dict

# ==================== 项目内部 ====================
# BaseNode: 导入流程节点基类
# StateFieldError: 状态字段类型/缺失异常
# ImportGraphState: 导入图状态
# generate_embeddings: 调用 BGE-M3 生成稠密/稀疏向量
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    # 节点在图中的注册名
    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        # ---------- 主流程 ----------
        # 步骤1：校验 state 中的 chunks
        chunks = self._step_1_validate_path(state)
        # 步骤2：批量调用 BGE，写入 dense/sparse 向量
        output_data = self._step_generate_embedding(chunks)

        # 先跑通看结果：暂不落盘备份
        # 步骤3：把带向量的 chunks 备份成 JSON
        self._step_backup_chunks(state, output_data)

        # 写回 state，供后续入库等节点使用
        state["chunks"] = output_data
        return state

    def _step_1_validate_path(self, state: ImportGraphState) -> List[Dict]:
        # 校验 chunks：必须存在，且类型为 list
        print("node_bge_embedding:步骤1:参数校验")
        chunks = state.get("chunks")

        # 空值（None / []）直接报错
        if not chunks:
            raise ValueError("参数错误:chunks")

        # 类型必须是 list
        if not isinstance(chunks, list):
            raise StateFieldError(
                field_name="chunks",
                message="chunks数据类型不正确",
                expected_type=list,
            )
        return chunks

    def _step_generate_embedding(self, chunks: List[Dict]) -> List[Dict]:
        # 按 batch 生成稠密向量 + 稀疏向量，写回每条 chunk
        print("node_bge_embedding:步骤2:数据向量化")
        output_data = []
        # 每批最多 5 条，控制显存/请求体大小
        batch_size = 5

        # 步长为 batch_size 滑动取批
        for i in range(0, len(chunks), batch_size):
            # 本批原始文档（最多 5 条）
            five_text = chunks[i:i + batch_size]
            # 本批待向量化的纯文本列表
            five_ready_xlh_texts = []
            for doc in five_text:
                item_name = doc.get("item_name", "") or ""
                content = doc.get("content", "") or ""
                # 商品名前置，强化检索特征
                five_ready_xlh_texts.append(
                    f"{item_name}\n{content}" if item_name else content
                )

            # 一次调用：返回 dense / sparse 两套向量
            embedding = generate_embeddings(five_ready_xlh_texts)

            # 按索引把向量挂回对应文档（copy 避免改原对象）
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
        # 解析备份目录（md 同目录 或 file_dir）
        backup_dir = self._resolve_backup_dir(state)
        # 文件名：{file_title}_new_new_chunks.json
        file_title = state.get("file_title") or "unknown"
        path = backup_dir / f"{file_title}_new_new_chunks.json"
        # 写出 JSON：保留中文、缩进 2
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        self.logger.info(f"加入稠密稀疏向量后的文档已备份: {path}")
        return str(path)

    @staticmethod
    def _resolve_backup_dir(state) -> Path:
        # 优先用 md_path 的父目录；否则用 file_dir；都没有则报错
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


# ==================== 本地调试入口 ====================
if __name__ == "__main__":
    node = NodeBGEEmbedding()
    # 先用 d 节点已有切片跑通（暂不依赖 e 的备份文件）
    with open(r"D:\output\hak180\hak180_new_chunks.json", "r", encoding="utf-8") as f:
        chunks_content = f.read()

    # 解析为 list[dict]，组装最小 state
    json_state = json.loads(chunks_content)
    init_state = {
        "file_title": "hak180_new",
        "md_path": r"D:\output\hak180\hak180_new.md",
        "file_dir": r"D:\output\hak180",
        "chunks": json_state,
    }

    # 执行节点（BaseNode 可调用）
    response = node(init_state)
    chunks = response.get("chunks", [])
    print(f"向量化完成，共 {len(chunks)} 条")
    # 打印第 1 条向量摘要，确认维度正常
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
