# 导入 json 模块：用于最后把结果序列化成 JSON 文本
import json
# 导入 logging 模块：用于输出日志
import logging
# 从 typing 导入类型标注：Any 任意类型，Dict 字典，List 列表
from typing import Any,Dict, List

# 导入 pydantic 的 AnyUrl 和 schema（本文件并未直接使用，保留原有导入）
from pydantic import AnyUrl, schema
# 导入 pymilvus 的 DataType（字段数据类型）和 milvus_client 模块
from pymilvus import DataType, milvus_client

# 导入 Milvus 配置对象：从配置里读取集合名称等参数
from config.milvus_config import milvus_config
# 导入基类 BaseNode（统一节点入口/日志/异常处理）和日志初始化函数 setup_logging
from processor.import_processor.base import BaseNode, setup_logging
# 导入自定义异常：StateFieldError 表示状态字段错误，MilvusError 表示 Milvus 操作错误
from processor.import_processor.exceptions import StateFieldError, MilvusError
# 导入流程图状态类型 ImportGraphState：描述节点之间传递的数据结构
from processor.import_processor.state import ImportGraphState
# 导入工具函数：get_milvus_client 获取客户端，escape_milvus_string 转义字符串
from utils.milvus_utils import get_milvus_client, escape_milvus_string


# 定义“导入 Milvus”节点类，继承 BaseNode，复用基类的日志与异常包装
class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """
    # 节点名称：日志里用它标识当前节点（基类据此创建 logger）
    name = "node_import_milvus"

    # 节点核心方法：由基类 __call__ 调用，输入和输出都是流程图状态
    def process(self, state: ImportGraphState):
        # 第一步：校验输入数据，返回（切片列表，向量维度）
        chunks_json_data,vector_dimension=self._step_1_check_inputs(state)

        # 第二步：连接 Milvus 并确保集合存在，返回 Milvus 客户端
        milvus_client=self._step_2_preare_collection(vector_dimension)

        # 第三步：按 file_title 删除同一文件的旧数据，避免重复导入
        self._self_3_clean_old_data(milvus_client,chunks_json_data)

        # 第四步：把切片插入 Milvus，并把返回的主键回填到 chunk_id
        updata_chunks=self._step_4_insert_data(milvus_client,chunks_json_data)
        # 遍历插入后的切片，逐条查看结果
        for chunk in updata_chunks:
            # 打印当前切片的 chunk_id（Milvus 主键），方便调试
            print(f"chunk_id:{chunk['chunk_id']}")

        # 把带 chunk_id 的切片写回状态，供后续节点或调用方使用
        state["chunks"]=updata_chunks

        # 返回更新后的状态（LangGraph 节点约定）
        return state

    # 步骤1：数据校验。检查 chunks 是否存在、是否非空、是否含 dense_vector
    def _step_1_check_inputs(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        # 打印步骤提示，方便在控制台跟踪执行进度
        print("node_import_milvus:步骤1:数据校验")

        # 从状态中取出上一步（向量化节点）产出的切片列表
        chunks_json_data=state.get("chunks")

        # 如果 chunks 为空（None 或空列表），说明没有可导入的数据，直接报错
        if not chunks_json_data:
            # 抛出状态字段异常：字段名 chunks、说明不能为空、期望类型 list
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        # 读取第一条切片的稠密向量长度，作为 Milvus 集合的向量维度
        demenstion=len(chunks_json_data[0]["dense_vector"])
        # 打印向量维度，便于确认是否与集合定义一致
        print(f"向量维度:{demenstion}")

        # 再次从状态取出 chunks（与上面的取值重复，保留原有逻辑）
        chunks = state.get("chunks")
        # 再次校验：chunks 不能为空，而且必须是 list 类型
        if not chunks or not isinstance(chunks, list):
            # 内容或类型不合法时抛出 ValueError
            raise ValueError("参数错误: chunks")

        # 取出第一条切片的稠密向量，确认向量化节点已经执行过
        dense_vector = chunks[0].get("dense_vector")
        # 如果 dense_vector 缺失或为空，说明还没跑向量化节点
        if not dense_vector:
            # 抛出 ValueError，提示先执行 f 节点（向量化）
            raise ValueError("参数错误: dense_vector 缺失，请先跑 f 节点向量化")

        # 返回切片列表和向量维度：维度用于创建集合时定义向量字段
        return chunks, len(dense_vector)

    # 步骤2：结构准备。连接 Milvus，集合不存在就按向量维度创建
    def _step_2_preare_collection(self, vector_dimension:int):
        # 打印步骤提示
        print("node_import_milvus:步骤2:结构准备")

        # 获取 Milvus 客户端连接（工具函数内部会读取配置）
        milvus_client=get_milvus_client()

        # 从配置中读取存放切片的集合名称
        collection_name=milvus_config.chunks_collection
        # 判断该集合是否已经存在
        if not milvus_client.has_collection(collection_name):
            # 不存在就创建集合（字段定义 + 向量索引）
            self.create_chunks_collection(milvus_client,collection_name,vector_dimension)

        # 返回客户端，后续步骤复用同一个连接
        return milvus_client

    # 步骤3：清理旧数据。按 file_title 删除同一文件的旧切片，实现“覆盖式”导入
    def _self_3_clean_old_data(self, client, chunks_json_data):
        # 打印步骤提示
        print("node_import_milvus:步骤3:清理旧数据")

        # 取第一条切片的 file_title，作为本次导入文件的标识
        file_title = chunks_json_data[0].get("file_title")
        # 用 try/except 包住删除操作，统一转换成 MilvusError
        try:
            # 转义字符串里的特殊字符，避免过滤表达式语法错误
            file_title = escape_milvus_string(file_title)
            # 执行删除：删掉这个文件标题下的所有旧数据
            client.delete(
                # 指定要删除数据的集合
                collection_name=milvus_config.chunks_collection,
                # 过滤条件：file_title 等于当前文件标题
                filter=f"file_title=='{file_title}'")
        # 捕获删除过程中的任何异常
        except Exception as e:
            # 用节点 logger 记录错误日志
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            # 抛出 MilvusError，交给基类统一包装处理
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")

    # 步骤4：插入数据。把切片写入 Milvus，并把主键回填到每个切片
    def _step_4_insert_data(self, client, chunks_json_data:List[Dict[str,Any]]) ->List[Dict[str,Any]]:
        # 打印步骤提示
        print("node_import_milvus:步骤4:插入数据")

        # 准备一个列表，用来装要插入的数据
        data_to_insert=[]
        # 遍历每一条切片，做插入前的字段补全
        for item in chunks_json_data:
            # 复制一份切片，避免修改原始数据
            item_copy=item.copy()
            # 如果这份切片里没有 part 字段
            if "part" not in item_copy:
                # 就补一个默认值 0，保证与集合字段定义匹配
                item_copy["part"]=0
            # 把处理好的副本加入待插入列表
            data_to_insert.append(item_copy)

        # 调用 Milvus 批量插入，collection_name 指定目标集合
        insert_result=client.insert(collection_name=milvus_config.chunks_collection,data=data_to_insert)
        # 从返回结果里取出实际插入的条数
        insert_count=insert_result.get("insert_count")
        # 打印插入条数，便于核对
        print(f"插入数据条数:{insert_count}")

        # 从返回结果里取出 Milvus 自动生成的主键列表（顺序与插入数据一致）
        insert_ids=insert_result.get("ids")
        # 遍历原始切片列表，按下标把主键写入对应切片
        for index,item in enumerate(chunks_json_data):
            # Milvus 主键是整数，转成字符串方便后续使用
            item["chunk_id"]=str(insert_ids[index])

        # 返回带 chunk_id 的切片列表
        return chunks_json_data

    # 创建切片集合：定义字段结构，以及稠密/稀疏向量的索引
    def create_chunks_collection(self,milvus_client,collection_name,vector_dimension):
        # 创建 schema（表结构）：auto_id=True 主键由 Milvus 自动生成；enable_dynamic_field 允许写入未定义字段
        schema=milvus_client.create_schema(auto_id=True,enable_dynamic_field=True)
        # 主键字段 chunk_id：INT64 类型，is_primary 标记为主键，auto_id 自动递增
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 切片正文字段 content：字符串类型，最大长度 65535
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        # 切片标题字段 title：字符串类型，最大长度 100
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
        # 父标题字段 parent_title：字符串类型，最大长度 100
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
        # 分片编号字段 part：整数类型 INT8
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        # 源文件标题字段 file_title：字符串类型，最大长度 100
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
        # 商品名称字段 item_name：字符串类型，最大长度 100（幂等性依据）
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
        # 稀疏向量字段 sparse_vector：稀疏浮点向量类型
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        # 稠密向量字段 dense_vector：浮点向量类型，维度由入参 vector_dimension 决定
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

        # 创建索引参数对象（用来登记下面要建的索引）
        index_params=milvus_client.prepare_index_params()

        # 再次创建索引参数对象（与上一行重复，保留原有逻辑）
        index_params=milvus_client.prepare_index_params()
        # 给稠密向量添加索引
        index_params.add_index(
            # 索引建在 dense_vector 字段上
            field_name="dense_vector",
            # 索引名称
            index_name="dense_vector_index",
            # 索引类型：AUTOINDEX 由 Milvus 自动选择
            index_type="AUTOINDEX",
            # 相似度度量方式：COSINE 余弦相似度
            metric_type="COSINE"
            )

        # 给稀疏向量添加索引
        index_params.add_index(
            # 索引建在 sparse_vector 字段上
            field_name="sparse_vector",
            # 索引名称
            index_name="sparse_inverted_index",
            # 索引类型：稀疏倒排索引
            index_type="SPARSE_INVERTED_INDEX",
            # 相似度度量方式：IP 内积
            metric_type="IP",
            # 索引额外参数：倒排算法 DAAT_MAXSCORE、向量归一化、不做量化
            params={"inverted_index_algo":"DAAT_MAXSCORE","normalize":True,"quantization":"none"}
            )
        # 创建集合：传入集合名、字段结构和索引参数
        milvus_client.create_collection(collection_name=collection_name,schema=schema,index_params=index_params)

# 主程序入口：直接运行本文件时，用本地 JSON 文件调试导入流程
if __name__ == "__main__":
    # 初始化日志（当前被注释掉，需要输出日志时可取消注释）
    # setup_logging()

    # 指定本地切片 JSON 文件的路径（含向量化结果）
    json_path = r"D:\output\hak180\hak180_new_new_new_chunks.json"
    # 以 UTF-8 编码打开文件
    with open(json_path, "r", encoding="utf-8") as f:
        # 一次性读出文件全部内容
        state_json = f.read()

    # 把 JSON 文本解析成 Python 对象（这里是一个切片列表）
    state = json.loads(state_json)

    # 组装成流程图状态：把切片放到 chunks 字段里
    init_state = {
        "chunks": state
    }

    # 执行核心处理流程
    # 创建节点实例
    node_import_milvus = NodeImportMilvus()
    # 调用节点（走基类 __call__），执行完整的导入流程
    result = node_import_milvus.process(init_state)

    # 把最终结果以 JSON 格式打印到日志，ensure_ascii=False 保留中文
    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
