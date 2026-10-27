# 第 6 章 f 节点：批处理与拷贝

`f_node_bge_embedding.py` 做的事很纯粹：把每个 chunk 拼成「商品名 + 正文」，分批送给 BGE-M3，拿回稠密和稀疏两套向量，挂回 chunk 上。它展示了三个非常常用的技巧：**步长批处理、切片取批、浅拷贝**。

## 6.1 range 的步长批处理

```python
# f_node_bge_embedding.py:66-69
batch_size = 5
for i in range(0, len(chunks), batch_size):
    five_text = chunks[i:i + batch_size]
```

**概念**：`range(start, stop, step)` 生成从 start 到 stop（不含）的等差数列。第三个参数是步长。

**示例**：`len(chunks) = 12`、`batch_size = 5` 时，`i` 依次为 0、5、10，三批分别是 `[0:5]`、`[5:10]`、`[10:15]`——最后一批只有 2 条，切片自动截断，不会越界。

**要点**：

- 这是「分批处理」的标准写法：按步长取起点，再用切片取一批。
- `math.ceil(n / batch_size)` 算批数，或直接 `for i in range(0, n, batch_size)`，后者更简洁。
- 批大小要兼顾吞吐与显存/请求体限制：太小网络往返多，太大容易 OOM 或被服务端拒绝。
- 也可以用 `itertools.islice` 或第三方库 `more_itertools.chunked`，但内置写法零依赖。

## 6.2 切片取批的边界

```python
five_text = chunks[i:i + batch_size]
```

**要点**：

- 切片越界永远安全：`[10:15]` 在长度 12 的列表上只返回 2 个元素。
- 切片返回**新列表**，但元素仍是原对象的引用，修改元素内容会影响原列表。
- 空切片 `[]` 不会报错；如果允许空批，后面要判空，否则 `generate_embeddings([])` 可能报错。

## 6.3 浅拷贝：copy() 到底复制了什么

```python
# f_node_bge_embedding.py:86-90
for j, doc in enumerate(five_text):
    item = doc.copy()
    item["dense_vector"] = embedding["dense"][j]
    item["sparse_vector"] = embedding["sparse"][j]
    output_data.append(item)
```

**概念**：`dict.copy()` 是**浅拷贝**——新建一个字典，把原字典的键值对「引用」复制过去。值如果是可变对象（list、dict），新旧字典仍共享同一个对象。

**示意**：

```python
a = {"tags": ["x"], "title": "t"}
b = a.copy()
b["title"] = "new"      # 不影响 a
b["tags"].append("y")   # 会影响 a！因为 tags 是同一个列表
```

**为什么这里用 copy**：

- 每个 chunk 要挂上自己的 `dense_vector` 和 `sparse_vector`，不能污染上游 `chunks`；
- chunk 的值大多是字符串（不可变），浅拷贝足够，比 `deepcopy` 快得多；
- `output_data` 是全新列表，原 `chunks` 保持不变。

**易错点**：如果 chunk 里有嵌套的可变字段（比如 `metadata` 字典），浅拷贝后修改 `metadata` 仍会互相影响，此时必须 `copy.deepcopy`。

## 6.4 列表构建与推导式

```python
# f_node_bge_embedding.py:71-80
five_ready_xlh_texts = []
for doc in five_text:
    item_name = doc.get("item_name", "") or ""
    content = doc.get("content", "") or ""
    five_ready_xlh_texts.append(
        f"{item_name}\n{content}" if item_name else content
    )
```

**等价的推导式**：

```python
five_ready_xlh_texts = [
    f"{doc.get('item_name') or ''}\n{doc.get('content') or ''}".strip()
    if (doc.get("item_name") or "") else (doc.get("content") or "")
    for doc in five_text
]
```

**要点**：

- 推导式更紧凑，但嵌套条件可读性差；本段用显式循环反而更清楚。
- `list.append` 在循环里逐个添加，均摊 O(1)；不要用 `list = list + [x]`，那会每次新建列表，O(n²)。
- `doc.get("item_name", "") or ""` 做了两层兜底：键不存在给空串，键存在但值为 None 也给空串。
- 三元表达式 `A if cond else B` 先算 cond，再选一个分支求值，不会两边都算。

## 6.5 enumerate：下标与元素同时拿

```python
for j, doc in enumerate(five_text):
    ...
    item["dense_vector"] = embedding["dense"][j]
```

**要点**：

- `enumerate(iterable)` 返回 `(下标, 元素)` 二元组，默认从 0 开始；`start=1` 可改起点。
- 这里必须用下标对齐：`embedding["dense"]` 是按输入顺序返回的列表，第 j 个输入对应第 j 个向量。
- 不要用 `for j in range(len(lst)): lst[j]`，既不 Pythonic 也容易越界。

## 6.6 类型校验与异常选择

```python
# f_node_bge_embedding.py:46-58
chunks = state.get("chunks")
if not chunks:
    raise ValueError("参数错误:chunks")
if not isinstance(chunks, list):
    raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)
```

**要点**：

- 同样是校验，这里抛 `ValueError`，e、g 节点抛 `StateFieldError`，风格不统一。项目内应统一用自定义异常，方便上层按类型捕获与展示。
- `ValueError` 是内置异常，语义是「参数值不合法」，用在这里不算错，只是丢失了字段名等上下文。
- 校验顺序建议：存在 → 非空 → 类型 → 元素结构，逐层收紧。

## 6.7 结果预览与长度

```python
# f_node_bge_embedding.py:152-160
c0 = chunks[0]
dense = c0.get("dense_vector") or []
sparse = c0.get("sparse_vector") or {}
print(f"第1条 dense 维度: {len(dense)}")
print(f"第1条 dense 前5维: {dense[:5]}")
print(f"第1条 sparse 非零项数: {len(sparse)}")
```

**要点**：

- `len(dense)` 是向量维度（如 1024），`len(sparse)` 是非零项个数，两者含义不同。
- `dense[:5]` 只打印前 5 维，避免刷屏；调试向量时这是好习惯。
- `c0.get("dense_vector") or []` 兜底后 `len(None)` 不会崩。
- 用 `json.dumps(summary, ensure_ascii=False, indent=2)` 打印摘要列表，比逐行 print 更整齐。

## 6.8 备份文件命名

```python
# f_node_bge_embedding.py:106-111
file_title = state.get("file_title") or "unknown"
path = backup_dir / f"{file_title}_new_new_chunks.json"
```

**要点**：

- 命名沿革：d 节点产出 `xxx_chunks.json`，e 节点产出 `xxx_new_chunks.json`，f 节点产出 `xxx_new_new_chunks.json`。层层加 `_new`，能看出处理阶段，但文件会越来越多。
- 更好的做法是「同一目录 + 阶段后缀」或「版本号」，例如 `xxx.chunks.v3.json`，并在配置里统一命名规则。
- `state.get("file_title") or "unknown"` 防止文件名变成 `None_new_new_chunks.json`。

## 6.9 本章问题清单

| 位置 | 现状 | 风险 | 建议 |
| --- | --- | --- | --- |
| 38 行 | 注释说「暂不落盘备份」，实际调用了备份 | 注释与代码不一致 | 更新注释 |
| 50 行 | 抛内置 ValueError | 与项目异常体系统一性差 | 改 StateFieldError |
| 87 行 | 浅拷贝 | 嵌套可变字段会串改 | 视结构用 deepcopy |
| 99-101 行 | 文件名层层 _new | 难以清理与识别 | 统一命名规范 |
| 115-128 行 | 与 e 节点重复的 _resolve_backup_dir | 复制粘贴 | 抽到基类或工具模块 |

# 第 7 章 g 节点：内置泛型、主键与幂等入库

`g_node_import_milvus.py` 是流水线的最后一站：校验 chunks、确保集合存在、按文件标题删旧数据、批量插入、回填主键。它的知识点偏「工程规范」：内置泛型、Any、异常转换、幂等写入。

## 7.1 PEP 585：内置类型也能下标

```python
# g_node_import_milvus.py:58
def _step_1_check_inputs(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
```

**概念**：Python 3.9 起，内置容器类型 `list`、`dict`、`tuple`、`set` 可以直接下标做泛型，不再必须从 `typing` 导入 `List`、`Dict`、`Tuple`。

```python
# 3.9+ 两种写法等价
def f(x: list[int]) -> dict[str, int]: ...
from typing import List, Dict
def f(x: List[int]) -> Dict[str, int]: ...
```

**要点**：

- 本行混用了两种风格：返回注解用内置 `tuple[...]`，元素里又用 `typing.List`。能跑，但风格不统一，建议全部改成内置泛型（3.9+）。
- `tuple[A, B]` 表示「二元组，第一项 A，第二项 B」，与 `Tuple[A, B]` 等价。
- `list[...]` 在 3.8 会 `TypeError`，所以库代码要兼容旧版本时仍用 `typing`。
- 注解依旧是「文档」，不影响运行。

## 7.2 Any：关闭类型检查的开关

```python
from typing import Any, Dict, List
def _step_4_insert_data(self, client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
```

**要点**：

- `Any` 表示「任意类型」，与任何类型都兼容，静态检查器遇到它就放弃检查。
- `Dict[str, Any]` 表示「键是字符串，值随便」，适合 chunk 这种字段多、类型不一的字典。
- 滥用 `Any` 等于放弃类型系统的保护；能用 `TypedDict` 精确描述 chunk 结构时，优先用 `TypedDict`。

## 7.3 冗余取值与命名拼写

```python
# g_node_import_milvus.py:63-90（节选）
chunks_json_data = state.get("chunks")
if not chunks_json_data:
    raise StateFieldError(...)

demenstion = len(chunks_json_data[0]["dense_vector"])
print(f"向量维度:{demenstion}")

chunks = state.get("chunks")
if not chunks or not isinstance(chunks, list):
    raise ValueError("参数错误: chunks")

dense_vector = chunks[0].get("dense_vector")
if not dense_vector:
    raise ValueError("参数错误: dense_vector 缺失，请先跑 f 节点向量化")

return chunks, len(dense_vector)
```

**要点**：

- 同一个 `state.get("chunks")` 取了两次、校验了两遍，属于复制粘贴留下的冗余；保留一份即可。
- 变量名 `demenstion` 拼写错误（dimension），且返回值又用 `len(dense_vector)` 重算了一次，`demenstion` 实际没被使用。
- 方法名 `_step_2_preare_collection`（prepare 少了个 p）、`_self_3_clean_old_data`（应为 `_step_3`）都是拼写/编号问题，不影响运行但影响维护。
- 先 `len(chunks_json_data[0]["dense_vector"])` 再判空，顺序反了：如果 `dense_vector` 缺失，这里会先抛 `KeyError`，而不是给出「请先跑 f 节点」的友好提示。应先校验再取值。

## 7.4 字典复制与字段补全

```python
# g_node_import_milvus.py:140-150
data_to_insert = []
for item in chunks_json_data:
    item_copy = item.copy()
    if "part" not in item_copy:
        item_copy["part"] = 0
    data_to_insert.append(item_copy)
```

**要点**：

- 用 `item.copy()` 避免给原 chunk 补 `part`，因为原数据可能还要用于别处或备份。
- `"part" not in item_copy` 是成员判断，比 `item_copy.get("part") is None` 更准确（能区分「键不存在」和「值为 None」）。
- 补默认值是为了匹配 Milvus schema 的 `part INT8` 字段：缺字段时 Milvus 会报错。
- 浅拷贝同样只复制一层，chunk 里的向量列表仍共享；插入后不会再改它们，所以安全。

## 7.5 主键回填

```python
# g_node_import_milvus.py:153-167
insert_result = client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
insert_count = insert_result.get("insert_count")
print(f"插入数据条数:{insert_count}")

insert_ids = insert_result.get("ids")
for index, item in enumerate(chunks_json_data):
    item["chunk_id"] = str(insert_ids[index])
```

**要点**：

- `MilvusClient.insert` 返回一个 dict，包含 `insert_count` 和 `ids`；`ids` 的顺序与插入列表一致，才能按下标回填。
- `enumerate` 同时拿到下标和元素，`insert_ids[index]` 与 `chunks_json_data[index]` 一一对应。
- 主键是 INT64，这里转成 `str` 存回 chunk，是为了 JSON 序列化和跨语言传输不丢精度（JS 的 Number 只有 53 位安全整数）。
- **注意**：回填的是本地 chunk 的 `chunk_id`，Milvus 里的字段类型仍是 INT64，两者只在 Python 侧有类型差异，不影响已插入的数据。

**易错点**：如果 `insert` 只插入了部分数据（异常中断），`ids` 长度可能小于输入长度，`insert_ids[index]` 会 `IndexError`。生产代码应校验 `len(insert_ids) == len(data_to_insert)`。

## 7.6 异常转换与异常链

```python
# g_node_import_milvus.py:118-132
try:
    file_title = escape_milvus_string(file_title)
    client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"file_title=='{file_title}'")
except Exception as e:
    self.logger.error(f"Milvus 数据删除失败: {str(e)}")
    raise MilvusError(f"Milvus 数据删除失败: {str(e)}")
```

**概念**：捕获底层异常（网络错误、SDK 异常），转换成项目自定义异常（`MilvusError`），让上层只处理一种异常类型。

**要点**：

- `raise ... ` 在 except 里重新抛，默认会保留原始异常上下文，traceback 里显示「During handling of the above exception, another exception occurred」。
- 想显式建立因果链，用 `raise MilvusError(...) from e`，traceback 会显示「The above exception was the direct cause」。
- 转换时把 `str(e)` 拼进消息，保留原始信息；但不要丢掉 `e` 本身，必要时用 `from e` 或挂到自定义异常的 `cause` 属性上。
- 只捕获 `Exception`，不要裸 `except:`，否则 Ctrl+C 也会被转成 MilvusError。

## 7.7 schema、索引与幂等：与 e 节点的对照

g 节点为 chunk 建的表比 e 节点的商品名表字段多得多：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| chunk_id | INT64 主键 auto_id | Milvus 自动生成 |
| content | VARCHAR(65535) | 切片正文 |
| title | VARCHAR(100) | 切片标题 |
| parent_title | VARCHAR(100) | 父标题 |
| part | INT8 | 分片序号 |
| file_title | VARCHAR(100) | 源文件 |
| item_name | VARCHAR(100) | 商品名（幂等依据） |
| sparse_vector | SPARSE_FLOAT_VECTOR | 稀疏向量 |
| dense_vector | FLOAT_VECTOR(dim) | 稠密向量，维度来自入参 |

**索引差异**：

- e 节点稠密索引用 `IVF_FLAT` + `nlist=128`；g 节点用 `AUTOINDEX`，由 Milvus 自动选型，省心但可控性弱。
- 两者稀疏索引都是 `SPARSE_INVERTED_INDEX` + `IP`，参数一致；索引名不同（`sparse_vector_index` 与 `sparse_inverted_index`），不影响功能。
- g 节点第 193、196 行连续调了两次 `prepare_index_params()`，第一次的结果被覆盖，是无用代码。

**幂等设计**：

```python
client.delete(collection_name=..., filter=f"file_title=='{file_title}'")
client.insert(collection_name=..., data=data_to_insert)
```

- 按 `file_title` 删除同一文件的旧切片，再插入新切片，实现「重复导入不重复」。
- 这与 e 节点按 `item_name` 删除是同一思路，但对象不同：商品名表按商品去重，切片表按文件去重。
- 过滤表达式里的字符串用单引号包裹，e 节点用双引号包裹，两种写法 Milvus 都支持，项目内应统一。
- `escape_milvus_string` 必须先于拼接执行；如果 `file_title` 为 None，转义函数可能返回 "None"，导致误删或删不到，应先校验。

**加载集合**：

```python
client.load_collection(collection_name)
```

- Milvus 的集合需要 load 到查询节点内存后才能 search/query；insert 本身不要求 load（部分版本的 delete 也要求集合已加载，以所用版本为准）。
- 每次导入都 load 一次，重复调用通常幂等，但会有开销；可以在服务启动时统一加载。

## 7.8 __main__ 调试入口

```python
# g_node_import_milvus.py:226-252
if __name__ == "__main__":
    json_path = r"D:\output\hak180\hak180_new_new_new_chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()
    state = json.loads(state_json)
    init_state = {"chunks": state}
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)
    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
```

**要点**：

- 用 JSON 文件模拟上游输出，是「单节点可复现调试」的好方法。
- `init_state` 只放了 `chunks`，没有 `file_title`；代码从 `chunks[0]["file_title"]` 取值，所以能跑，但这是隐式依赖。
- `json.loads` 把字符串解析成 Python 对象；`json.load` 则直接从文件读。这里先 `read()` 再 `loads()`，多了一步，可以改成 `json.load(f)`。
- `logging.getLogger().info(...)` 用的是根 logger，需先 `setup_logging()` 才有格式；本文件把它注释掉了，所以输出是默认格式。

## 7.9 本章问题清单

| 位置 | 现状 | 风险 | 建议 |
| --- | --- | --- | --- |
| 9、11 行 | pydantic 的 AnyUrl、schema 与 milvus_client 未使用 | 无用依赖 | 删除 |
| 39 行 | 局部变量与导入模块同名 milvus_client | 遮蔽，易误读 | 改名 client |
| 42 行 | `_self_3_clean_old_data` | 拼写/编号错误 | 改名 _step_3 |
| 58 行 | tuple[...] 与 typing.List 混用 | 风格不一致 | 统一 |
| 63-90 行 | 重复取值与校验 | 冗余 | 合并 |
| 71 行 | 先取 dense_vector 再判空 | 可能 KeyError | 先校验 |
| 193、196 行 | prepare_index_params 调用两次 | 后者覆盖前者 | 删除一次 |
| 231 行 | 硬编码调试路径 | 换环境即失效 | 用参数或配置 |

# 第 8 章 横向串讲：按主题重排的知识地图

前七章按文件顺序讲，便于对照代码；这一章把同样的知识按主题重新归类，适合复习时查漏补缺。

## 8.1 文件、路径与 IO

| 知识点 | 关键 API | 出现位置 |
| --- | --- | --- |
| 路径对象 | Path、/、parent、name、stem、suffix | a、b、c、d、e、f |
| 存在性判断 | exists、is_file、is_dir | a、b、c |
| 目录创建 | mkdir(parents=True, exist_ok=True) | b |
| 重命名 | with_name、rename | b |
| 文本读 | open(..., "r", encoding="utf-8")、read | b、d、e、f、g |
| 文本写 | open(..., "w", encoding="utf-8")、write | c、d、e、f |
| 二进制读 | open(..., "rb") | b、c |
| 二进制写 | open(..., "wb") | b |
| 目录列举 | os.listdir、Path.iterdir | c |
| 扩展名 | os.path.splitext、Path.suffix | a、c |
| 压缩包 | zipfile.ZipFile、extractall | b |

**记忆点**：**读文本要编码，读二进制不要；路径拼接用 `/`，不要用字符串加。**

## 8.2 字符串与正则

| 知识点 | 关键 API | 出现位置 |
| --- | --- | --- |
| 格式化 | f-string、`:.2f` | 全文 |
| 去空白 | strip、lstrip、rstrip | c、d、e |
| 大小写 | lower | a、c |
| 前后缀 | startswith、endswith | c、d |
| 替换 | replace、re.sub | c、d、e |
| 切分与拼接 | split、join | d、e |
| 编译 | re.compile | c |
| 查找 | search、match、span | c、d |
| 转义 | re.escape | c |
| 非贪婪 | `.*?` | c |
| 字符类与量词 | `\s`、`\d`、`{}`、`+`、`*` | c、d |

**记忆点**：**字符串不可变，所有方法都返回新串；拼正则必转义，跨行匹配加 re.S。**

## 8.3 容器、解包与迭代

| 知识点 | 关键 API | 出现位置 |
| --- | --- | --- |
| 列表增删 | append、extend、pop、remove、切片 | c、d、e、f、g |
| 字典读写 | get、items、in、赋值、copy | 全文 |
| 元组 | 多返回值、解包、`_` 占位 | b、c、e |
| 嵌套结构 | list[dict]、tuple 嵌套 | c、e、f、g |
| 枚举 | enumerate(start=1) | d、e、f、g |
| 成员判断 | in、not in | d、g |
| 排序与推导式 | 列表推导、条件表达式 | f、g |

**记忆点**：**`append` 加一个、`extend` 摊平一批；`in` 对 list 是 O(n)，对 dict 是 O(1)。**

## 8.4 函数、闭包与面向对象

| 知识点 | 关键 API | 出现位置 |
| --- | --- | --- |
| 函数定义 | def、默认参数、关键字实参 | 全文 |
| 多返回值 | return a, b；元组解包 | b、e、g |
| 嵌套函数 | 闭包、晚绑定 | d |
| 匿名函数 | lambda | c |
| 类与继承 | class X(Base)、self、类属性 | a-g |
| 方法重写 | process | a-g |
| 模板方法 | 基类 __call__ 调子类 process | base.py |
| 抽象基类 | ABC、@abstractmethod | base.py |
| 静态方法 | @staticmethod | e、f |
| 魔术方法 | __call__、__init__、__enter__、__exit__、__str__ | base、b、exceptions |

**记忆点**：**self 不是关键字，是约定；继承只写差异；闭包捕获变量而非值。**

## 8.5 异常与资源管理

| 知识点 | 关键 API | 出现位置 |
| --- | --- | --- |
| 抛出 | raise、raise ... from e | a、b、c、d、e、g |
| 捕获 | try/except/else/finally | b、c、g |
| 自定义异常 | ImportProcessError 体系 | exceptions.py、a、b、e、g |
| 异常转换 | 捕获底层异常再抛业务异常 | g |
| 上下文管理器 | with open、with ZipFile | b、c、d、e、f、g |
| 自定义上下文 | __enter__ / __exit__ | base（概念） |

**记忆点**：**早失败、晚捕获；捕获具体异常；finally 一定执行；with 保证释放。**

## 8.6 序列化、日志与类型

| 知识点 | 关键 API | 出现位置 |
| --- | --- | --- |
| JSON 序列化 | json.dumps、json.dump | b、c、d、e、f、g |
| JSON 反序列化 | json.loads、json.load | e、f、g |
| 中文保留 | ensure_ascii=False | b、d、e、f、g |
| 日志 | logging.getLogger、info、warning、error | 全文 |
| 日志配置 | basicConfig、format | base.py |
| 类型注解 | 参数/返回值、List/Dict/Tuple、内置泛型 | 全文 |
| 类型描述 | TypedDict、total=False、Any | state.py、g |

**记忆点**：**dump 进文件、dumps 出字符串；日志用模块级 logger，不要 print。**

## 8.7 外部生态

| 领域 | 库 | 在本项目中的用途 |
| --- | --- | --- |
| HTTP | requests | 调 MinerU、上传/下载文件 |
| PDF 解析 | MinerU（HTTP 服务） | PDF → Markdown |
| 大模型 | langchain_deepseek、langchain_core | 商品名识别、图片摘要 |
| 文本切分 | langchain_text_splitters | 递归字符切分 |
| 向量化 | BGE-M3（项目封装） | dense + sparse |
| 向量库 | pymilvus | 建表、索引、增删查 |
| 对象存储 | minio | 图片上传、公网 URL |
| 配置 | dotenv + dataclass | 环境变量加载 |

# 第 9 章 代码问题总表与改进建议

把七个文件里值得改的地方集中列出，按严重程度排序。这张表同时是一份「Python 代码审查清单」。

## 9.1 问题总表

| 严重度 | 位置 | 问题 | 后果 | 建议 |
| --- | --- | --- | --- | --- |
| 高 | b:49-60 | 裸 except 吞异常后使用未赋值变量 | 真实错误被掩盖，最终报 NameError | 捕获具体异常并 raise，或提前初始化 |
| 高 | b:10、c:36-38 | 未使用的 Django 导入 | 环境缺 Django 时 import 即失败 | 删除 |
| 高 | c:104 | `state["md_content"]` 直接下标 | 缺键抛 KeyError | 改 get 并校验 |
| 高 | d:186 | `Path(state.get("md_path"))` | md_path 为 None 时 TypeError | 先校验 |
| 高 | e:376-396 | delete 后 insert，无事务 | insert 失败会丢旧数据 | 先写新再删旧，或用版本字段 |
| 中 | b:133-181 | 轮询写在 with open 内 | 文件句柄长时间占用 | 上传后立即退出 with |
| 中 | b:183 | 不可达的 return | 死代码误导 | 删除 |
| 中 | c:147、304 | 循环内重复 compile | 性能浪费 | 提前编译 |
| 中 | c:263-265 | 先删旧图再上传 | 上传失败导致图片丢失 | 先上传后清理 |
| 中 | d:8-15 | 多个未使用导入 | 依赖风险、可读性差 | 删除 |
| 中 | d:186、209 | 变量遮蔽 sys.path/sys.prefix | 可读性差、潜在误用 | 改名 |
| 中 | e:172 | 每次新建 LLM 客户端 | 连接不复用 | 用缓存工厂 |
| 中 | e:376 | 索引只在建表时创建 | 已有集合缺索引 | 单独检查索引 |
| 中 | g:42、93 | 方法名拼写错误 | 维护困难 | 重命名 |
| 中 | g:63-90 | 重复取值与校验 | 冗余、逻辑分叉 | 合并 |
| 低 | a:35-41 | suffix 未转小写 | .PDF 不支持 | lower() |
| 低 | a:48 | 输出目录硬编码 | 换环境失效 | 读配置 |
| 低 | b:141-147 | end_time 命名与含义不符 | 误读 | 改 elapsed |
| 低 | c:219 | 创建未使用的客户端 | 资源浪费 | 删除 |
| 低 | c:353 | Windows 路径未加 r | 转义告警 | 加 r |
| 低 | d:143 | 打印整个 sections | 日志爆炸 | 只打数量 |
| 低 | f:38 | 注释与代码不符 | 误导 | 更新注释 |
| 低 | g:193-196 | prepare_index_params 调用两次 | 无效代码 | 删除一次 |

## 9.2 如果只改三处

1. **修 b 节点的异常处理**（最高优先级）。吞掉异常再抛 NameError，会让「文件读失败」这种简单问题排查半天。改成捕获 `OSError`/`UnicodeDecodeError` 并重新抛出，或直接不捕获。
2. **清理所有未使用导入**。Django 的三处导入是定时炸弹：换一台没装 Django 的机器，节点直接 import 失败。用 `ruff check --select F401` 可以自动发现。
3. **把硬编码路径与限流参数改成配置**。`D:\output`、`D:\hak180.pdf`、`max_requests=10` 散落在代码里，换环境就要改代码。统一走 `ImportConfig` 与环境变量。

## 9.3 顺带补上的 Python 版本知识

| 特性 | 最低版本 | 本项目中的例子 |
| --- | --- | --- |
| f-string | 3.6 | 全文 |
| 数据类 dataclass | 3.7 | import_config.py |
| 海象运算符 `:=` | 3.8 | c:297 |
| TypedDict | 3.8（typing） | state.py |
| 内置泛型 `list[int]` | 3.9 | g:58 |
| 字典合并运算符（两个竖线） | 3.9 | 未使用 |
| 结构化模式匹配 `match` | 3.10 | 未使用 |
| 联合类型（竖线分隔，如 `str` 与 `None`） | 3.10 | 未使用 |
| 异常组 `except*` | 3.11 | 未使用 |

**结论**：项目实际要求 **Python 3.9+**（因为 g 节点用了 `tuple[...]`）。如果环境是 3.8，g 节点会在导入时（准确说是函数定义时）报 `TypeError: 'type' object is not subscriptable`。

# 第 10 章 学习路径与自测

## 10.1 建议的学习顺序

| 阶段 | 内容 | 目标 |
| --- | --- | --- |
| 第 1 步 | 第 1 章 a 节点 | 能独立写出「继承 BaseNode + process + 校验 + 返回 state」的节点 |
| 第 2 步 | 第 2 章 b 节点 | 会发 HTTP 请求、读写文件、用 with、写 try/except |
| 第 3 步 | 第 3、4 章 c、d 节点 | 掌握字符串、正则、切片、闭包、JSON |
| 第 4 步 | 第 5、6 章 e、f 节点 | 理解类型注解、引用与拷贝、批处理 |
| 第 5 步 | 第 7 章 g 节点 | 能设计 Milvus schema、索引，写出幂等入库 |
| 第 6 步 | 第 8、9 章 | 把知识串成体系，能 review 别人的代码 |

## 10.2 自测题

1. `state.get("x")` 与 `state["x"]` 有什么区别？分别在什么场景用？
2. 为什么 b 节点在读文件失败后，第 60 行会抛 `NameError`？如何避免？
3. `with open(...) as f:` 背后的两个魔术方法是什么？异常时文件会被关闭吗？
4. `re.escape` 解决什么问题？不用它会怎样？
5. `deque.popleft()` 与 `list.pop(0)` 的复杂度差别？为什么限流器选 deque？
6. 浅拷贝 `dict.copy()` 在什么情况下不够用？举例说明。
7. d 节点的 `_flush_section` 为什么能读到循环中不断变化的 `current_lines`？
8. `@staticmethod` 与普通实例方法的核心区别？什么时候该用？
9. e 节点给 `chunk["item_name"]` 赋值，为什么 `state["chunks"]` 里也能看到？
10. 为什么 Milvus 要先 delete 再 insert？这样做的风险是什么？
11. 裸 `except:` 与 `except Exception:` 有什么区别？各自的风险？
12. `tuple[List[Dict], int]` 这种写法需要哪个 Python 版本？3.8 上会怎样？

## 10.3 参考答案要点

1. `get` 缺键返回 None，`[]` 缺键抛 KeyError。外部输入、可选字段用 get；内部约定必存在的字段用 `[]` 让错误早暴露。
2. `open` 失败后 `md_content` 从未赋值，except 又把它吞了，执行到赋值语句就 NameError。应让异常上抛或在 try 前初始化。
3. `__enter__` 返回资源，`__exit__` 负责释放；即使 with 块内抛异常，`__exit__` 也会被调用，文件一定关闭。
4. 把文件名里的 `.`、`(`、`?` 等正则元字符转义成字面量。不转义会导致匹配错误甚至正则语法错误。
5. `popleft` 是 O(1)，`pop(0)` 是 O(n)（要移动后续元素）。限流器每次请求都要清理队首，必须 O(1)。
6. 值里有嵌套可变对象（list/dict）时，浅拷贝后两边仍共享，改一个会影响另一个；此时用 `copy.deepcopy`。
7. 闭包捕获的是变量（cell），不是值；`_flush_section` 在循环中被调用时读取的是当时的绑定，所以是最新值。若延迟到循环结束后调用，会全部读到最后一个值。
8. 静态方法没有 `self`/`cls`，不能访问实例或类状态，靠参数工作；逻辑与实例无关、可独立测试时用。
9. `chunks` 是列表，元素是 dict；循环变量 `chunk` 是 dict 的引用，改它等于改原对象，列表里存的就是这些对象。
10. 用「先删后插」模拟 upsert，保证同商品不重复。风险是两步非原子，insert 失败会丢旧数据；应改为先写新版本再删旧版本，或用唯一键 + 版本字段。
11. 裸 except 捕获 BaseException，连 Ctrl+C、SystemExit 都吞；`except Exception` 只捕获常规异常，保留中断语义。
12. 需要 3.9+（PEP 585）。3.8 上函数定义时就会 `TypeError: 'type' object is not subscriptable`。

# 附录 A 运行环境与调试方式

## A.1 依赖与版本

| 类别 | 依赖 | 用途 |
| --- | --- | --- |
| 运行时 | Python 3.9+（建议 3.10/3.11） | 内置泛型、海象运算符 |
| HTTP | requests | 调 MinerU |
| 大模型 | langchain-deepseek、langchain-core | 文本与多模态调用 |
| 切分 | langchain-text-splitters | 递归字符切分 |
| 向量库 | pymilvus | Milvus 客户端 |
| 对象存储 | minio | 图片上传 |
| 配置 | python-dotenv | 加载 .env |
| Web | django（被误引入） | 实际未使用，可删 |

## A.2 单节点调试

每个文件末尾都有 `if __name__ == "__main__":`，可以直接运行：

```bash
python processor/import_processor/nodes/a_node_entry.py
python processor/import_processor/nodes/d_node_document_split.py
```

调试前确认：

- `D:\output` 目录存在（a、b 节点会校验）；
- `.env` 里配好 MinerU token、LLM key、Milvus 地址、MinIO 账号；
- 上游节点产出的 JSON 文件路径与 `__main__` 里写的一致。

## A.3 调试小技巧

- 用 `python -X dev` 运行，可以看到被默认忽略的 `SyntaxWarning`（比如 Windows 路径的无效转义）。
- 打印 state 时用 `json.dumps(state, ensure_ascii=False, indent=2, default=str)`，`default=str` 能兜住 Path 等不可序列化对象。
- 在 `BaseNode.__call__` 里加计时日志，能快速定位是哪个节点慢。
- 用 `logging.basicConfig(level=logging.DEBUG)` 打开调试级别，`self.logger.debug` 的合并细节才会输出。

# 附录 B 术语表

| 术语 | 含义 |
| --- | --- |
| state | 流水线各节点共享的字典，承载输入、中间结果与输出 |
| 节点（node） | 继承 BaseNode、实现 process 的一个处理单元 |
| 模板方法模式 | 父类定义流程骨架（__call__），子类实现具体步骤（process） |
| TypedDict | 用类语法描述字典结构的类型工具，运行时不校验 |
| 闭包 | 内层函数引用外层函数变量形成的绑定 |
| 晚绑定 | 闭包在调用时才读取变量当前值，而非定义时的值 |
| 上下文管理器 | 实现 __enter__/__exit__、可被 with 使用的对象 |
| 海象运算符 | `:=`，在表达式内部赋值并返回该值 |
| 浅拷贝 | 只复制容器本身，元素仍共享引用 |
| 深拷贝 | 递归复制所有层级，完全独立 |
| 幂等 | 同一操作执行多次，结果与执行一次相同 |
| 轮询 | 间隔固定时间反复查询任务状态，直到终态 |
| 稠密向量 | 固定维度的浮点数组，表达语义 |
| 稀疏向量 | 只有少量非零维度的向量，表达关键词权重 |
| 混合检索 | 稠密与稀疏检索结合，再融合排序 |
| 预签名 URL | 带临时授权的上传/下载链接，客户端可直连存储 |
| data URL | `data:image/jpeg;base64,...` 形式的内联资源 |
| RAG | 检索增强生成：先检索知识，再交给大模型回答 |

---

**文档结束**。建议阅读顺序：先看第 0 章建立全局观，再按 a→g 过一遍代码，最后用第 8、9 章复习与自查。
