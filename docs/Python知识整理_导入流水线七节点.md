# 第 0 章 总览：七个节点与知识地图

## 0.1 这条流水线在做什么

这七个文件是同一条「文档导入流水线」的七个环节：把一个 PDF（或 Markdown）文件，变成可被检索的知识库数据。数据以 `state` 字典为载体，在节点之间单向流动，每个节点只做一件事，并把结果写回 `state`：

- a 节点：判断文件类型，决定走 PDF 还是 MD 分支；
- b 节点：调用 MinerU 把 PDF 解析成 Markdown；
- c 节点：给 Markdown 里的图片生成摘要、上传 MinIO、替换图片链接；
- d 节点：按标题把长文切成 chunk，过长再切、过短再合；
- e 节点：用大模型识别商品名，并把商品名向量化后写入 Milvus；
- f 节点：对每个 chunk 批量生成稠密 + 稀疏向量；
- g 节点：把带向量的 chunk 写入 Milvus，并回填主键。

因此，这七份代码其实覆盖了一条相当完整的 Python 工程链路：**语法 → 文件与路径 → 文本与正则 → 网络请求 → 面向对象与类型注解 → 第三方生态（LLM、向量库、对象存储）**。

## 0.2 节点清单与知识分布

| 顺序 | 节点文件 | 输入 → 输出 | 外部能力 | 主要知识点 |
| --- | --- | --- | --- | --- |
| a | a_node_entry.py | import_file_path → file_title、file_dir、读取开关 | 无 | 导入、类与继承、Path、异常、f-string |
| b | b_node_pdf_to_md.py | pdf_path、file_dir → md_content、md_path | MinerU HTTP API | requests、文件 IO、with、try/except、轮询、zipfile |
| c | c_node_md_img.py | md_path、md_content → 新 md_content、md_path | DeepSeek 多模态、MinIO | 正则、字符串、deque 限流、base64、lambda、海象运算符 |
| d | d_node_document_split.py | md_content、file_title → chunks | LangChain 切分器 | 容器、闭包、状态机、re.match、json |
| e | e_node_item_name_recognition.py | chunks、file_title → item_name、chunks | DeepSeek、BGE-M3、Milvus | 类型注解、校验、静态方法、引用语义、向量库建模 |
| f | f_node_bge_embedding.py | chunks → 带向量的 chunks | BGE-M3 | 步长批处理、切片、浅拷贝、enumerate |
| g | g_node_import_milvus.py | chunks → 带 chunk_id 的 chunks | Milvus | 内置泛型、Any、异常转换、主键回填、幂等入库 |

## 0.3 知识难度梯度

| 阶段 | 节点 | 知识主题 | 难度 |
| --- | --- | --- | --- |
| 地基 | a | 模块、类、继承、类型注解、字典、异常、Path | 入门 |
| 输入输出 | b | HTTP、文件读写、上下文管理器、异常处理、轮询 | 入门到中级 |
| 文本处理 | c、d | 字符串、正则、切片、闭包、容器、JSON | 中级 |
| 数据建模 | e、f | 类型系统、引用与拷贝、批处理、向量概念 | 中级到进阶 |
| 外部存储 | g | Milvus schema、索引、幂等写入、主键 | 进阶 |

## 0.4 阅读约定

- 行号均基于当前仓库版本，格式为「文件名:行号」，例如 `a_node_entry.py:29`。
- 代码片段为节选，省略号表示与当前知识点无关的代码。
- 每节固定三段式：先讲概念，再看代码位置，最后说「为什么这么写 / 易错在哪里」。
- 第 1 至 7 章按 a→g 的**代码顺序**讲；第 8 章按**主题**横向重排，便于复习；第 9 章集中列出代码中值得改进的地方。

> 提示：阅读时不必按顺序背 API。先理解「state 在节点间流动」这条主线，再把每个节点用到的语法当成工具箱里的工具，用到时能想起来即可。

# 第 1 章 a 节点：Python 语法地基

`a_node_entry.py` 只有 51 行，却把后面所有节点的公共骨架都用了一遍：模块导入、类继承、类型注解、字典读写、异常、路径、f-string。读懂它，等于拿到了整条流水线的「最小可运行模型」。

## 1.1 模块、包与 import

**概念**：一个 `.py` 文件就是一个模块；一个带 `__init__.py` 的目录就是一个包。`import` 做了三件事：找到模块文件、执行模块顶层代码、把模块对象缓存进 `sys.modules`。因此同一个模块在进程里只会被执行一次，之后再导入只是取缓存。

```python
# a_node_entry.py:1-7
import logging
from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
```

**要点**：

- `import logging` 绑定的是「模块对象」，用的时候写 `logging.info(...)`。
- `from pathlib import Path` 绑定的是「模块里的名字」，用的时候直接写 `Path(...)`。
- `from ... import A, B` 一次导入多个名字，逗号分隔，不是两个 import。
- 这四行里既有标准库（logging、pathlib），也有本项目模块（processor.*），说明导入按「标准库 → 第三方 → 项目内」分层，这是 PEP 8 的推荐顺序。

**易错点**：包导入会执行 `__init__.py`；b 节点导入了 `django.utils.archive`，只要 Django 没装，整个文件在 import 阶段就会报 `ModuleNotFoundError`，根本走不到 `process`。这就是「未使用导入」的真实代价，详见 9.1 节。

## 1.2 类、继承与方法重写

**概念**：`class NodeEntry(BaseNode)` 表示 NodeEntry 继承 BaseNode，自动获得父类的属性和方法；子类只写与父类不同的部分，这叫「重写（override）」。

```python
# a_node_entry.py:9-19
class NodeEntry(BaseNode):
    """入口节点：任务分发"""

    name = "node_entry"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")
```

**要点**：

- `name` 写在类体里、没有 `self.`，是**类属性**，所有实例共享；基类用它拼 logger 名 `import.node_entry`。
- `process` 的第一个参数是 `self`，代表实例本身。调用时写 `node.process(state)`，Python 会自动把 `node` 传给 `self`，不需要手动传。
- 基类 `BaseNode` 里 `self.config`、`self.logger` 在 `__init__` 中创建，子类没有写 `__init__`，就自动继承父类的构造逻辑。
- 真正的入口不是 `process`，而是基类的 `__call__`：`node(state)` 会先打「开始」日志，再调用 `self.process(state)`，最后统一包一层 `ImportProcessError`。这叫**模板方法模式**——父类定流程，子类填内容。
- 基类同时继承 `ABC` 并把 `process` 标成 `@abstractmethod`，所以只继承不实现 `process` 的子类无法实例化，会在 `TypeError` 中直接暴露出来。

```python
# base.py:49-82（节选）
def __call__(self, state: T) -> T:
    try:
        self.logger.info(f"--- {self.name} 开始 ---")
        result = self.process(state)
        self.logger.info(f"--- {self.name} 完成 ---")
        return result
    except Exception as e:
        raise ImportProcessError(message=str(e), node_name=self.name, cause=e)
```

**为什么这样设计**：七个节点都需要日志、计时、异常包装。如果每个 `process` 各写一遍，既重复又容易漏。把公共逻辑提到 `__call__`，子类只关心业务，这是面向对象里「复用」最典型的用法。

**易错点**：d、e、g 三个文件的 `__main__` 里直接调用 `node.process(init_state)`，绕过了 `__call__`，所以看不到基类的开始/完成日志，也不会被包装成 `ImportProcessError`。调试时可以这样，但要清楚两者的区别。

## 1.3 函数定义与类型注解

```python
# a_node_entry.py:17
def process(self, state: ImportGraphState):
    ...
```

**概念**：`state: ImportGraphState` 是**类型注解**（type hint），它只是给人和 IDE 看的说明，**运行时不会检查**。传一个普通 dict 进去照样能跑，传一个字符串也不会被拦下。

`ImportGraphState` 来自 `state.py`，是一个 `TypedDict`：

```python
# state.py:16
class ImportGraphState(TypedDict, total=False):
    task_id: str
    file_title: str
    md_content: str
    chunks: List
    ...
```

**要点**：

- `TypedDict` 描述「一个字典应该有哪些键、值的类型是什么」，用于静态检查（mypy、PyCharm）和阅读，不产生运行时约束。
- `total=False` 表示所有键都是可选的：访问不存在的键在类型检查器看来是合法的，但运行时 `state["xxx"]` 仍会抛 `KeyError`。这正是后面多处用 `state.get()` 的原因。
- `process` 没有写返回值注解，从代码看它返回的是 state，规范写法应为 `-> ImportGraphState`。g 节点就写了返回注解，风格并不统一。

## 1.4 字典：state 的读写

**概念**：`state` 本质是一个 dict。读取有两种写法，行为完全不同：

| 写法 | 键不存在时 | 适用场景 |
| --- | --- | --- |
| `state["key"]` | 抛 KeyError | 确定键一定存在，希望缺失时立刻暴露 |
| `state.get("key")` | 返回 None | 键可能缺失，需要自己判断 |
| `state.get("key", 默认值)` | 返回默认值 | 需要兜底值 |

```python
# a_node_entry.py:22-31
import_file_path = state.get("import_file_path")
if not import_file_path:
    raise StateFieldError(field_name="import_file_path", expected_type=str)

import_file_path_obj = Path(import_file_path)
if not import_file_path_obj.exists():
    raise FileProcessingError(message=f"文件不存在:{import_file_path}")
```

**要点**：

- 先 `get` 再判空，是「防御式编程」的标准两行：取不到就报一个语义明确的业务异常，而不是让 `Path(None)` 抛出难懂的 TypeError。
- 字典是**可变对象**，`state["is_md_read_enabled"] = True` 是原地修改，函数内不需要 `global`，调用方也能看到变化。
- 节点最后 `return state`，既改了原对象又返回它，两种约定都满足；LangGraph 更推荐「返回新状态或更新后的状态」，避免隐式副作用。

## 1.5 真值测试与空值判断

```python
# a_node_entry.py:25
if not import_file_path:
    ...
```

**概念**：Python 里每个对象都有「真值」。以下值在 `if` 中为假：

- `None`、`False`、数值 `0`、`0.0`
- 空字符串 `""`、空列表 `[]`、空元组 `()`、空字典 `{}`、空集合 `set()`

**要点**：

- `if not x` 同时覆盖了「没这个键（None）」和「键存在但为空串」两种情况，对路径校验正好合适。
- 如果业务上「0 是合法值」，就必须写 `if x is None`，否则会被误判。判断 None 永远用 `is`，不要用 `==`。
- 反过来，`if x:` 是「非空且非零」的简写，后文大量出现，例如 f 节点 `item_name = doc.get("item_name", "") or ""`。

## 1.6 异常：raise 与自定义异常

**概念**：异常是「中断正常流程、把错误向上传递」的机制。`raise` 抛出异常对象，最近的 `except` 捕获它；没人捕获就一路向上，最终打印 traceback 并退出。

```python
# a_node_entry.py:25-32
if not import_file_path:
    raise StateFieldError(field_name="import_file_path", expected_type=str)
...
if not import_file_path_obj.exists():
    raise FileProcessingError(message=f"文件不存在:{import_file_path}")
```

**要点**：

- 项目自定义了一套异常层次（`exceptions.py`）：`ImportProcessError` 是基类，下面分 `StateFieldError`、`FileProcessingError`、`PdfConversionError`、`MilvusError` 等。抛业务异常而不是裸 `Exception`，调用方才能按类型分别处理。
- `StateFieldError(field_name=..., expected_type=str)` 用的是**关键字实参**：参数名写出来，顺序无关，可读性更好。它的构造函数在 message 为空时会自动拼出「状态字段 'x' 缺失或无效，期望类型: str」。
- 「早失败（fail fast）」：路径不合法就在入口处报错，别等到上传文件时才炸。a 节点只做校验和分发，不碰业务逻辑。
- 异常会一路向上，被 `BaseNode.__call__` 捕获后重新包装成 `ImportProcessError`，并保留 `cause` 原始异常。

**易错点**：`raise XxxError("msg")` 与 `raise XxxError(message="msg")` 取决于构造函数签名。`FileProcessingError` 直接继承 `ImportProcessError`，第一个位置参数就是 message，所以两种都行；`StateFieldError` 的第一个位置参数是 `node_name`，写成位置参数就会把消息塞进 node_name，必须用关键字。

## 1.7 pathlib：面向对象的路径

```python
# a_node_entry.py:29-47
import_file_path_obj = Path(import_file_path)
if not import_file_path_obj.exists():
    ...
if import_file_path_obj.suffix == ".md":
    state["is_md_read_enabled"] = True
elif import_file_path_obj.suffix == ".pdf":
    state["is_pdf_read_enabled"] = True
    state["pdf_path"] = import_file_path
...
state["file_title"] = import_file_path_obj.stem
state["file_dir"] = r"D:\output"
```

**概念**：`pathlib.Path` 把路径封装成对象，用属性和运算符代替字符串拼接，天然跨平台。

| 写法 | 含义 | 示例结果 |
| --- | --- | --- |
| `Path(p)` | 构造路径对象 | Path("D:/data/a.pdf") |
| `.exists()` | 是否存在（文件或目录） | True / False |
| `.is_file()` / `.is_dir()` | 是文件 / 是目录 | True / False |
| `.name` | 最后一段（含后缀） | "a.pdf" |
| `.stem` | 最后一段（不含后缀） | "a" |
| `.suffix` | 后缀（含点） | ".pdf" |
| `.parent` | 上级目录 | Path("D:/data") |
| `p / "sub"` | 拼接路径 | Path("D:/data/sub") |
| `.with_name("b.md")` | 替换最后一段 | Path("D:/data/b.md") |
| `.rename(new)` | 重命名（返回新路径） | 文件被改名 |
| `.mkdir(parents=True, exist_ok=True)` | 建目录，父目录一起建，已存在不报错 | 目录就绪 |
| `.absolute()` | 绝对路径（不解析符号链接） | D:\data\a.pdf |
| `.resolve()` | 规范化绝对路径（解析链接与 ..） | 同上，更彻底 |

**要点**：

- `Path` 对象和字符串互转：`str(path)` 转回字符串，`Path("...")` 转成对象。b 节点最后 `return str(new_md_path.absolute())`，因为 state 里约定存字符串。
- 后缀判断要小心大小写：`"A.PDF"` 的 suffix 是 `".PDF"`，a 节点用 `== ".pdf"` 会漏判，稳妥写法是 `suffix.lower()`。
- `r"D:\output"` 是**原始字符串**，让反斜杠不被当作转义符。Windows 路径建议一律加 `r` 前缀，或直接用正斜杠 `"D:/output"`。

**易错点**：c 节点的 `"D:\output\hak180\hak180.md"` 没有加 `r`，`\o`、`\h` 不是合法转义，Python 会给出 SyntaxWarning（未来版本会直接报错）；更危险的是 `\n`、`\t`、`\b` 这类组合，它们会真的变成换行、制表符、退格，把路径悄悄改坏。Windows 路径统一加 `r` 前缀最稳。

## 1.8 字符串与 f-string

```python
# a_node_entry.py:19
logging.info(f"{self.name}节点开始执行...")
# a_node_entry.py:32
raise FileProcessingError(message=f"文件不存在:{import_file_path}")
```

**要点**：

- `f"..."` 在字符串里用 `{}` 嵌入表达式，可放变量、属性、函数调用、甚至运算。
- 格式控制用冒号：`f"{duration:.2f}"`（保留两位小数，c 节点限流日志用过）、`f"{n:>5}"`（右对齐）。
- `f"{self.name}节点"` 里 `self.name` 是属性访问，不是方法调用。
- 字符串比较用 `==`，不要用 `is`；`is` 比较的是对象身份，只用于 None、True、False 等单例。

**易错点**：f-string 的大括号里不能出现反斜杠（3.12 之前），也不能嵌套同种引号。c 节点拼接长提示词时用三引号 f-string，变量插值后缩进会原样保留，发给模型前最好用 `textwrap.dedent` 或手动对齐。

## 1.9 本章小结

| 知识点 | 关键 API / 语法 | 在 a 节点中的位置 |
| --- | --- | --- |
| 模块导入 | import / from ... import | 1-7 行 |
| 类与继承 | class X(Base), 类属性, self | 9-17 行 |
| 类型注解 | `state: ImportGraphState` | 17 行 |
| 字典读写 | get / [] / 赋值 | 22-48 行 |
| 真值判断 | if not x | 25 行 |
| 异常 | raise 自定义异常 | 26、32、44 行 |
| 路径 | Path, exists, suffix, stem | 29-47 行 |
| 字符串 | f-string, 原始字符串 | 19、32、47、48 行 |

**一句话**：a 节点 = 用 Path 判断类型，用 dict 传递状态，用异常表达「输入不合格」。

> 易错点清单：① `suffix == ".pdf"` 未处理大小写；② Windows 路径未加 `r`；③ md 分支只设了 `is_md_read_enabled`，没有设置 `md_path`，后续 c 节点会因缺字段失败；④ 输出目录硬编码为 `D:\output`，应走配置。

# 第 2 章 b 节点：HTTP、文件与异常

`b_node_pdf_to_md.py` 是整条链路里「对外交互」最重的一环：把 PDF 上传给 MinerU，轮询解析进度，下载 zip，解压，改文件名，读回 Markdown。它把 **网络请求、文件 IO、上下文管理器、异常处理、循环控制** 一次性用全了。

## 2.1 导入结构与「导入即执行」

```python
# b_node_pdf_to_md.py:1-15
import json
import logging
import time
import zipfile
from http.client import responses
from pathlib import Path
from urllib.parse import urlsplit

import requests
from django.utils.archive import extract

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState
```

**要点**：

- 标准库、第三方库、项目内模块分三组，组间空行，这是 PEP 8 的导入顺序约定。
- `json` 只在 `__main__` 里用；`logging`、`time`、`zipfile`、`Path` 在正文用；`responses`、`urlsplit`、`extract` 全文未使用。
- **导入即执行**：`import` 会执行被导入模块的顶层语句。`from django.utils.archive import extract` 会让本文件强依赖 Django——即便一次都没用到 `extract`，Django 不在环境里也会在 import 阶段失败。这是「删掉无用导入」不只是洁癖的原因。

**易错点**：`from http.client import responses` 里的 `responses` 是 HTTP 状态码字典（如 `{200: "OK", ...}`），不是常用工具；真正的响应对象来自 `requests`。名字撞车容易误导阅读。

## 2.2 requests：一次完整的 HTTP 交互

这个节点与 MinerU 的交互分三步，正好覆盖三种 HTTP 方法：

```python
# b_node_pdf_to_md.py:110-136（节选）
header = {
    "Content-type": "application/json",
    "Authorization": f"Bearer {api_token}"
}
data = {"files": [{"name": pdf_path_obj.name}], "model_version": "vlm"}

url = f"{base_url}/file-urls/batch"
response = requests.post(url, headers=header, json=data)
if response.status_code != 200:
    raise FileProcessingError(message=f"申请上传文件失败:{response.text}")
result = response.json()
...
with open(pdf_path_obj, "rb") as pdf_file:
    res_upload = requests.put(signed_urls, data=pdf_file)
```

**概念与参数**：

| 参数 | 作用 |
| --- | --- |
| `url` | 请求地址，f-string 拼出完整路径 |
| `headers=` | 请求头字典，`Authorization: Bearer <token>` 是常见鉴权格式 |
| `json=` | 把 Python 对象序列化成 JSON 放进请求体，并自动加 Content-Type |
| `data=` | 原始请求体；传文件对象时会把文件内容作为 body 上传 |
| `timeout=` | 超时秒数，不设就可能永久阻塞（轮询处设了 10 秒） |
| `response.status_code` | HTTP 状态码，200 表示成功 |
| `response.text` | 响应体文本，报错时打印它便于排查 |
| `response.json()` | 把响应体解析成 Python 对象（dict / list） |
| `response.content` | 响应体字节，下载二进制文件时用 |

**要点**：

- `requests.post(url, headers=..., json=...)` 返回 `Response` 对象，**不会**因为 4xx/5xx 自动抛异常，必须自己判断 `status_code`。若想自动抛，可用 `response.raise_for_status()`。
- 业务层还有一层状态码：`result.get("code") != 0`。HTTP 200 只代表「请求送达」，不代表「业务成功」，两层都要判断。
- 上传用的是「预签名 URL」：先申请一个临时链接（`signed_urls`），再把文件 PUT 上去。这样文件不经过业务服务器，省带宽。
- 变量名 `signed_urls` 是复数，但取的是 `[0]`，单个 URL，命名可以更准确。

**易错点**：`requests.put(signed_urls, data=pdf_file)` 里如果传的是打开的文件对象，务必确保请求期间文件没被关闭——b 节点把整个轮询循环写在了 `with open(...)` 内部，文件句柄会被占住最长 10 分钟，这是不推荐的（见 2.5）。

## 2.3 嵌套字典与安全取值

```python
# b_node_pdf_to_md.py:126-130
result = response.json()
if result.get("code") != 0:
    raise FileProcessingError(message=f"申请上传文件失败:{result.get('message')}")
batch_id = result["data"]["batch_id"]
signed_urls = result["data"]["file_urls"][0]
```

**要点**：

- `response.json()` 返回的通常是嵌套 dict：`{"code":0, "data": {"batch_id": "...", "file_urls": [...]}}`。
- 链式下标 `result["data"]["batch_id"]` 简洁，但任何一层缺失都会 `KeyError`；`result.get("code")` 缺失时返回 None，`None != 0` 成立，反而能进入错误分支，这是「用 get 做协议判断」的技巧。
- 读列表用下标 `[0]`，读可能不存在的字段用 `.get(key, 默认值)`：`extract_result.get("err_msg", "未知错误,无具体信息")`。

**易错点**：不要对深层结构连续 `.get()` 而不判空，`result.get("data").get("batch_id")` 在 data 为 None 时会 `AttributeError`。安全写法是先取出中间层并判断，或用 `try/except KeyError`。

## 2.4 文件读写：模式与编码

```python
# b_node_pdf_to_md.py:133
with open(pdf_path_obj, "rb") as pdf_file:
    ...
# b_node_pdf_to_md.py:201-202
with open(zip_save_path, "wb") as f:
    f.write(response.content)
# b_node_pdf_to_md.py:50-51
with open(md_path, "r", encoding="utf-8") as md_file:
    md_content = md_file.read()
```

**模式速查**：

| 模式 | 含义 | 文件不存在时 | 写入位置 |
| --- | --- | --- | --- |
| `r` | 只读文本 | 报 FileNotFoundError | 不可写 |
| `w` | 只写文本 | 创建 | 清空重写 |
| `a` | 追加文本 | 创建 | 末尾追加 |
| `rb` / `wb` | 二进制读 / 写 | 同上 | 不做编码转换 |
| `r+` | 读写 | 报错 | 从开头覆盖 |

**要点**：

- 文本模式必须显式写 `encoding="utf-8"`，否则用系统默认编码（Windows 中文环境常是 GBK），读写中文会乱码或报 `UnicodeDecodeError`。
- 二进制模式不涉及编码，图片、zip、PDF 一律用 `rb` / `wb`。
- `.read()` 一次性读入内存，适合中小文件；大文件应分块 `for chunk in f:` 或 `read(size)`。
- `.write()` 的返回值是写入的字符数，一般不关心。

## 2.5 with 与上下文管理器

**概念**：`with open(...) as f:` 背后的协议是：进入时调用 `f.__enter__()` 返回 `f`，离开时无论正常结束还是异常都调用 `f.__exit__()` 关闭文件。所以文件一定会被关闭，不需要手动 `close()`。

```python
# b_node_pdf_to_md.py:208-209
with zipfile.ZipFile(zip_save_path, mode="r") as zip_file_obj:
    zip_file_obj.extractall(extract_target_dir)
```

**要点**：

- 自定义对象只要实现 `__enter__` / `__exit__` 就能用在 `with` 里，常见于锁、数据库连接、临时目录、HTTP 会话。
- 多个资源可以写成一个 with：`with open(a) as f, open(b) as g:`。
- b 节点的 `with open(pdf_path_obj, "rb")` 把轮询循环整个包住了，缩进层级深、文件句柄长期占用。更好的写法是：先在 with 里把文件读进内存或上传完，再退出 with 去轮询。

**易错点**：`with` 退出时如果 `__exit__` 返回 True，异常会被吞掉；文件对象的 `__exit__` 返回 None，所以异常照常抛出。

## 2.6 try / except / finally

```python
# b_node_pdf_to_md.py:49-56
try:
    with open(md_path, "r", encoding="utf-8") as md_file:
        md_content = md_file.read()
except:
    print("其实你的程序已经成功了!!!")
finally:
    print("over")
```

**执行顺序**：

1. 执行 try 块；
2. 没有异常 → 跳过 except，执行 finally，继续往下；
3. 有异常且匹配某个 except → 执行该 except，再执行 finally；
4. 有异常但不匹配 → 先执行 finally，再把异常向上抛。

**要点**：

- `finally` 一定执行，适合放清理代码（关连接、删临时文件、打印收尾日志）。
- `except:` 是「裸 except」，等价于 `except BaseException:`，会连 `KeyboardInterrupt`（Ctrl+C）、`SystemExit` 一起吞掉，调试时非常难受。至少写 `except Exception as e:`。
- 本段最严重的问题不是裸 except，而是**异常被吞后变量未定义**：`open` 失败时 `md_content` 从未赋值，第 60 行 `state["md_content"] = md_content` 会抛 `NameError`，错误信息与真实原因（文件读失败）完全无关。正确做法是让异常继续抛，或在 except 里 `raise`。

**易错点**：`except Exception as e:` 里的 `e` 只在 except 块内有效（Python 3 会在这个块结束时删除 `e`），出了块再引用会 `NameError`。

## 2.7 循环、轮询与时间控制

```python
# b_node_pdf_to_md.py:140-181（节选）
poll_url = f"{base_url}/extract-results/batch/{batch_id}"
start_time = time.time()
timeout_seconds = 600
poll_interval = 3

while True:
    end_time = time.time() - start_time
    if end_time > timeout_seconds:
        raise FileProcessingError(message="获得下载地址超时")
    try:
        res_poll = requests.get(url=poll_url, headers=header, timeout=10)
    except Exception as e:
        self.logger.error(f"轮询接口异常:{e}")
        time.sleep(poll_interval)
        continue

    if res_poll.status_code != 200:
        raise PdfConversionError(...)

    ...
    if extract_state == "done":
        full_zip_url = extract_result["full_zip_url"]
        return full_zip_url
    elif extract_state == "failed":
        ...
        raise PdfConversionError(...)
    else:
        self.logger.info(...)
        time.sleep(poll_interval)
```

**概念**：轮询（polling）就是「隔一段时间问一次，直到状态变成终态」。异步任务系统（视频转码、文档解析、模型训练）普遍采用这种模式。

**要点**：

- `while True` 是死循环，必须有出口：`return`（成功）、`raise`（失败或超时）。这里三条出口都齐全。
- `time.time()` 返回当前时间戳（秒，浮点），两者相减得到已耗时。注意它受系统时钟调整影响，测耗时更稳的是 `time.monotonic()`。
- 变量名 `end_time` 实际存的是「已耗时」，叫 `elapsed` 更准确。
- `continue` 跳过本轮剩余代码，直接进入下一轮；网络抖动时「记日志 + 睡 3 秒 + 继续」是合理的容错。
- `sleep` 会阻塞当前线程，这是**同步**模型；如果要同时处理多个任务，需要线程池或 asyncio。
- 超时预算 600 秒、间隔 3 秒，最多约 200 次请求；生产环境建议加「指数退避」（3s、6s、12s……）避免给服务端压力。

**易错点**：`if res_poll.status_code != 200: raise ...` 在轮询里直接抛，意味着「一次 5xx 就整体失败」，与上面 catch 网络异常继续重试的策略不一致。真实项目里 5xx 通常也应重试，只有 4xx（参数/鉴权错误）才立即失败。

## 2.8 多返回值与元组解包

```python
# b_node_pdf_to_md.py:39
pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)
# b_node_pdf_to_md.py:92
return pdf_path_obj, output_dir_obj
```

**概念**：`return a, b` 实际返回一个元组 `(a, b)`；接收端用同样数量的变量「解包」即可。这是 Python 最常用的多返回值方式，不需要定义额外的类。

**要点**：

- 返回 2 个值就声明 2 个变量，数量不匹配会 `ValueError: too many values to unpack`。
- 不关心的值用 `_` 占位：`for img_file, img_path, _ in target_images`（c 节点）。
- 星号解包：`first, *rest = [1, 2, 3]`，`rest` 得到 `[2, 3]`。
- 交换变量：`a, b = b, a`，底层就是元组打包与解包。

## 2.9 zipfile：解压

```python
# b_node_pdf_to_md.py:200-209
zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
with open(zip_save_path, "wb") as f:
    f.write(response.content)

extract_target_dir = output_dir_obj / pdf_stem
extract_target_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_save_path, mode="r") as zip_file_obj:
    zip_file_obj.extractall(extract_target_dir)
```

**要点**：

- `ZipFile(path, "r")` 读取 zip；`extractall(目录)` 全部解压；`extract(成员, 目录)` 解压单个；`namelist()` 列出成员。
- 标准库的 `extract` 会对成员名做一定清洗（去掉盘符、开头的 `/`、`..`），但仍建议只解压可信来源，并限制解压后大小，防止 zip 炸弹。
- `mkdir(parents=True, exist_ok=True)`：父目录不存在就一起建，目录已存在也不报错，是「确保目录存在」的幂等写法。

## 2.10 Path 的组合与重命名

```python
# b_node_pdf_to_md.py:214-222
target_md_file = extract_target_dir / "full.md"
new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
target_md_file.rename(new_md_path)
return str(new_md_path.absolute())
```

**要点**：

- `dir / "full.md"` 用斜杠运算符拼接，比 `os.path.join` 可读。
- `with_name("x.md")` 只替换最后一段，目录保持不变，比字符串 `replace` 安全。
- `rename` 在同一文件系统内是改名；Windows 上目标已存在会 `FileExistsError`，重跑前要先删或先判断。
- `absolute()` 返回绝对路径但不解析 `..` 和符号链接；需要规范化用 `resolve()`。

## 2.11 __main__ 自测块

```python
# b_node_pdf_to_md.py:225-241
if __name__ == "__main__":
    setup_logging()
    init_state = {"pdf_path": r"D:\hak180.pdf", "file_dir": r"D:\output"}
    node = NodePDFToMD()
    result = node(init_state)
    dumps = json.dumps(result, ensure_ascii=False, indent=4)
    print(dumps)
```

**概念**：每个模块都有 `__name__`。被直接运行时它是 `"__main__"`，被导入时它是模块名。所以这段代码只在「单独运行本文件」时执行，被别的模块 import 时不会跑。

**要点**：

- 这是给单个节点做「最小可运行测试」的标准写法：构造最小 state，实例化节点，调用，打印结果。
- `json.dumps(obj, ensure_ascii=False, indent=4)`：`ensure_ascii=False` 保留中文（默认会转成 `\uXXXX`），`indent=4` 缩进 4 空格便于阅读。
- `dumps` 是「对象 → 字符串」，`dump` 是「对象 → 文件」；`loads` / `load` 同理。四个函数名只差一个 s，别记混。
- `node(init_state)` 走基类 `__call__`，会打印开始/完成日志；直接 `node.process(...)` 则不会。

## 2.12 本章问题清单

| 位置 | 现状 | 风险 | 建议 |
| --- | --- | --- | --- |
| 5、7、10 行 | responses、urlsplit、extract 未使用 | 无谓依赖，Django 缺失时直接 ImportError | 删除 |
| 49-56 行 | 裸 except 吞异常 | 掩盖真实错误；后续 NameError | 捕获具体异常并 raise |
| 60 行 | 依赖 try 中赋值的变量 | 读文件失败时 NameError | 提前初始化或让异常上抛 |
| 133-181 行 | 轮询写在 with open 内 | 文件句柄长时间占用 | 上传完即退出 with |
| 141、146 行 | end_time 实为已耗时 | 命名误导 | 改名 elapsed |
| 183 行 | return "下载url" | 永远执行不到的死代码 | 删除 |
| 197 行 | 下载 zip 未设 timeout | 可能永久阻塞 | 加 timeout |
| 43、53、56 行 | print 调试 | 生产环境无级别、无时间 | 统一用 logger |

> 本章最值得记住的一句：**「HTTP 200 不等于业务成功，try 住异常不等于处理了异常。」**


# 第 3 章 c 节点：文本、正则与限流

`c_node_md_img.py` 的任务是「把 Markdown 里的本地图片换成摘要 + 公网链接」。它涉及的知识点最杂：导入陷阱、字符串处理、正则、切片、deque 限流、base64、多模态消息、lambda、海象运算符。这一章按代码出现顺序逐个拆解。

## 3.1 文件级 docstring 与注释

```python
# c_node_md_img.py:1-19
"""
【文件说明】c_node_md_img.py —— Markdown 图片处理节点
...
5 个处理步骤（对应下面 process 方法）：
    步骤1  读出 md 正文、md 路径、图片目录
    ...
"""
```

**要点**：

- 模块开头的三引号字符串是 **docstring**，可用 `help(模块)` 或 IDE 悬浮查看，也是自动生成文档的来源。
- docstring 与普通注释（`#`）不同：注释是给维护者看的，docstring 是对象 `__doc__` 属性的一部分。
- 本文件采用「文件头总述 + 每个方法 docstring + 行内注释」三层结构，可读性很好，值得沿用。

## 3.2 导入的真相：覆盖、无效与未使用

```python
# c_node_md_img.py:31-34
from ast import Tuple, List, Dict
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple
```

**概念**：Python 的 `import` 本质是「把名字绑定到对象」。后绑定的会覆盖先绑定的，且不报错。

**发生了什么**：

1. 第 31 行从 `ast` 导入 `Tuple`、`List`、`Dict`——它们是抽象语法树节点类，跟类型注解毫无关系；
2. 第 34 行从 `typing` 再导入同名对象，把前三个名字覆盖掉；
3. 最终生效的是 `typing` 的版本，所以代码能跑，但第 31 行完全是无效导入。

**为什么危险**：如果顺序反过来，`ast.Tuple` 就会生效，`List[Tuple[...]]` 会在运行时抛 `TypeError`（ast 节点类不支持下标）。这类 bug 极难排查。

```python
# c_node_md_img.py:36-38
from django.contrib.messages.constants import SUCCESS
from django.contrib.messages.context_processors import messages
from django.db.models.expressions import result
```

**要点**：

- 三个 Django 导入全文未使用，属于「无用依赖」；文件注释也承认本项目并未安装 Django。
- 只要环境里没有 Django，`import c_node_md_img` 就会失败，与是否调用无关。
- 判断导入是否必要，不能只看「能不能跑」，要看「删掉会不会影响功能」。

## 3.3 os.path 与 pathlib 混用

```python
# c_node_md_img.py:121-128
for image_file in os.listdir(images_dir):
    file_ext = os.path.splitext(image_file)[1].lower()
    if file_ext not in self.config.image_extensions:
        ...
    img_path = images_dir / image_file
```

**要点**：

- `os.listdir(dir)` 返回文件名列表（不含目录），只列一层，不递归；目录不存在会抛 `FileNotFoundError`。
- `os.path.splitext("a.tar.gz")` 返回 `("a.tar", ".gz")`，只拆最后一个点；`splitext("README")` 返回 `("README", "")`。
- `.lower()` 把扩展名统一成小写，避免 `.PNG` 被白名单漏掉——a 节点恰好缺了这一步。
- `images_dir` 本身是 `Path` 对象，所以还能用 `/` 拼接。同一个文件里 `os.path` 和 `pathlib` 混用虽然能工作，但风格不统一，建议统一到 `Path`：`for p in images_dir.iterdir()`。

## 3.4 字符串方法大盘点

这个文件几乎把常用字符串方法用了一遍：

| 方法 | 作用 | 代码位置 |
| --- | --- | --- |
| `.lower()` | 转小写 | 122 行 |
| `.strip()` | 去首尾空白（含换行、制表符） | 252 行 |
| `.replace("\n", "")` | 替换子串 | 252 行 |
| `.startswith()` | 是否以某前缀开头（d 节点） | d:126 行 |
| `.split("\n")` | 按分隔符切成列表（d 节点） | d:99 行 |
| `"\n".join(lines)` | 用分隔符把列表拼成字符串（d 节点） | d:115 行 |
| `.lstrip()` | 只去左侧空白 | d:214 行 |
| `f"{...}"` | 格式化 | 全文 |

**要点**：

- 这些方法都返回**新字符串**，原字符串不变（字符串不可变）。所以必须写 `md_content = md_content.replace(...)`，只写一行 `md_content.replace(...)` 是无效操作。
- `strip()` 不传参时去掉所有空白字符（空格、`\n`、`\t`、`\r`）；传参则去掉指定字符集合，例如 `.strip('"')`。
- `split()` 不传参按任意空白切分，并自动去掉空串；`split("\n")` 则保留空行（空行会得到空字符串元素），d 节点正需要保留行结构。

## 3.5 正则表达式：查找与替换

这是 c 节点的核心技能。目标：在 Markdown 里找到 `![alt](path/xxx.png)` 这样的图片语法。

```python
# c_node_md_img.py:147-160
pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
match = pattern.search(md_content)

if not match:
    return None

start, end = match.span()
context_len = self.config.img_content_length
pre_text = md_content[max(0, start - context_len):start]
post_text = md_content[end:min(len(md_content), end + context_len)]
```

**逐段拆解**：

| 片段 | 含义 |
| --- | --- |
| `!` | 字面量感叹号 |
| `\[` | 字面量左方括号（转义，否则是字符类） |
| `.*?` | 任意字符，非贪婪（尽可能少匹配） |
| `\]` | 字面量右方括号 |
| `\(` `\)` | 字面量圆括号 |
| `re.escape(image_file)` | 把文件名里的 `.`、`(` 等元字符转义成字面量 |

**API 说明**：

- `re.compile(pattern)` 预编译成正则对象，重复使用时比每次传字符串快（本文件在循环里每次 compile，其实可以提到循环外）。
- `.search(text)` 从任意位置找**第一个**匹配；`.match()` 只从开头匹配；`.fullmatch()` 要求整串匹配。
- `.span()` 返回匹配的 `(起, 止)` 下标，止是开区间，`text[start:end]` 正好是匹配内容。
- `.sub(repl, text)` 替换**所有**匹配；`repl` 可以是字符串、函数或 lambda。
- `re.escape` 是防止「文件名里的特殊字符」被当成正则语法，这是拼接正则时最容易忘的一步。

**为什么用非贪婪 `.*?`**：一张图后面可能还有别的图片语法，贪婪的 `.*` 会一路吃到最后一个 `)`，把整段都匹配进去。非贪婪保证匹配到最近的右括号。

**易错点**：

- 正则里的反斜杠建议用原始字符串 `r"..."`，否则 `\d`、`\s` 可能被 Python 先转义一次。
- `.` 默认不匹配换行，Markdown 图片语法通常在一行内，所以没问题；跨行匹配要加 `re.S`。
- 用字符串做 `sub` 的替换内容时，`\1`、`\g<name>` 会被当成组引用，反斜杠也会被解释；用 lambda 返回替换文本可以完全避开这些规则——c 节点第 305 行正是这么做的。

```python
# c_node_md_img.py:302-307
for image_file, (summary, url) in image_info.items():
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
    md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)
```

**要点**：`lambda m: ...` 接收 match 对象、返回替换字符串；这里 `m` 没用到，但仍必须保留这个参数。

## 3.6 切片与边界控制

```python
# c_node_md_img.py:156-158
context_len = self.config.img_content_length
pre_text = md_content[max(0, start - context_len):start]
post_text = md_content[end:min(len(md_content), end + context_len)]
```

**概念**：`s[a:b]` 取下标 `a` 到 `b-1`。越界不会报错：`a` 超过长度得到空串，`b` 超过长度自动截到末尾；但负数下标会从尾部倒数，所以**必须防负数**。

**要点**：

- `max(0, start - context_len)` 防止起点变负；`min(len(md_content), end + context_len)` 防止终点越界（其实 Python 自己会截断，这里写出来是显式表达意图）。
- 切片是**浅拷贝**：对字符串得到新字符串，对列表得到新列表（元素仍是同一批对象引用）。
- `s[::-1]` 反转，`s[::2]` 隔一个取一个，`s[:]` 复制整个列表。

## 3.7 嵌套容器与解包

```python
# c_node_md_img.py:137
target_images.append((image_file, img_path, context))
# c_node_md_img.py:170
for img_file, image_path, context in target_images:
# c_node_md_img.py:230-231
if image_content:
    pre_text, post_text = image_content
```

**数据结构**：`target_images` 是 `list[tuple[str, Path, tuple[str, str]]]`——列表里放三元组，三元组第三项又是二元组。

**要点**：

- `for a, b, c in 列表` 会自动解包每个元素；元素长度不匹配会 `ValueError`。
- `pre_text, post_text = image_content` 解包二元组；先判空再解包，避免 `TypeError: cannot unpack non-sequence NoneType`。
- 嵌套结构用类型注解写出来（第 162 行 `List[Tuple[str, str, Tuple[str, str]]]`），可读性会好很多，虽然注解本身不检查。

## 3.8 deque 与滑动窗口限流

**需求**：调用视觉模型时，限制「任意 60 秒内最多 10 次请求」，避免触发对方限流。

**朴素做法**：用 list 记录每次请求时间，每次判断前 `list.pop(0)`。问题：`pop(0)` 要把后面所有元素前移，复杂度 O(n)。

**正确做法**：`collections.deque` 双端队列，两端插入/删除都是 O(1)。

```python
# c_node_md_img.py:190-211（节选）
current_time = time.time()

while request_times and current_time - request_times[0] >= window_seconds:
    request_times.popleft()

if len(request_times) >= max_requests:
    sleep_duration = window_seconds - (current_time - request_times[0])
    if sleep_duration > 0:
        ...
        time.sleep(sleep_duration)
        current_time = time.time()
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

request_times.append(current_time)
```

**算法三步**：

1. **清理过期**：队首是最早的请求，只要它超出窗口就 `popleft()`，直到队首在窗口内；
2. **超限等待**：窗口内已有 `max_requests` 次，就算出「最早那次请求还有多久滑出窗口」，`sleep` 等它出去；
3. **记录本次**：把当前时间戳 `append` 到队尾，供下次判断。

**要点**：

- 队列里始终只保存「窗口内」的时间戳，长度不会无限增长。
- `deque(maxlen=N)` 可以自动丢弃旧元素，适合固定长度的滑动窗口；这里需要按时间判断，所以手动清理。
- `time.sleep` 会阻塞线程，限流器是「同步阻塞」实现；高并发场景要用令牌桶 + 异步或分布式限流（Redis）。
- **参数不一致**：调用处写死 `max_requests=10`（第 172 行），而配置里有 `requests_per_minute = 15`，两处应统一。

## 3.9 base64 与多模态消息

```python
# c_node_md_img.py:216-252（节选）
with open(image_path, "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"这是\"{root_folder}\"文件中的一张图片……"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    }
]
response = vl_ai.invoke(messages)
return response.content.strip().replace("\n", "")
```

**概念**：

- 图片是二进制，不能直接塞进 JSON 字符串；base64 把每 3 字节编码成 4 个 ASCII 字符，体积增大约 1/3。
- `b64encode` 接收 `bytes`，返回 `bytes`；`.decode("utf-8")` 转成 `str` 才能拼进 JSON。
- `data:image/jpeg;base64,<数据>` 是 **data URL**，浏览器和多数多模态模型都支持这种内联图片格式。
- 消息格式是多模态协议：`content` 不再是字符串，而是「内容块列表」，每块用 `type` 区分文本和图片。

**要点**：

- `open(..., "rb")` 读出的就是 `bytes`，不需要再 `.encode()`。
- `vl_ai.invoke(messages)` 是同步调用，返回消息对象，正文在 `.content`。
- 返回前 `.strip().replace("\n", "")` 去掉首尾空白和换行，方便直接当图片标题。
- 第 219 行创建了一个 `ChatDeepSeek` 实例 `ai`，但后面调用的是 `get_llm_client` 返回的 `vl_ai`，`ai` 完全没用到——又是一处无效代码。

**易错点**：`data:image/jpeg` 是写死的，PNG 图片也标成 jpeg。多数模型能自动识别，但严格来说应按扩展名或 MIME 类型生成。

## 3.10 lambda 与海象运算符

```python
# c_node_md_img.py:294-300
def merge_summery_and_url(self, summaries: Dict[str, str], urls) -> Dict[str, Tuple[str, str]]:
    image_info = {}
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    return image_info
```

**海象运算符 `:=`**（Python 3.8+）：在表达式里赋值，并返回赋的值。上面这行等价于：

```python
url = urls.get(image_file)
if url:
    ...
```

**要点**：

- 好处是少写一行，且赋值与判断挨在一起，语义清晰。
- 优先级低，复杂表达式里要加括号：`if (n := len(x)) > 10:`。
- 不能用在「只赋值不使用」的场合，它的设计目的就是「赋值后立刻用」。
- `lambda` 是匿名函数：`lambda m: f"![{summary}]({url})"` 等价于一个只 return 这一行的 def。适合作为 `sort(key=...)`、`map`、`sub` 的回调，逻辑复杂时应改回 def。

## 3.11 异常兜底：什么时候该吞

```python
# c_node_md_img.py:272-282
def clean_minio_directory(self, minio_client: Minio, upload_dir):
    try:
        objects_to_delete = minio_client.list_objects(...)
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
        if delete_list:
            errors = minio_client.remove_objects(...)
            for error in errors:
                self.logger.error(f"删除失败：{error}")
    except Exception as e:
        self.logger.error(f"清理minio目录失败：{e}")
```

**要点**：

- 这里「吞掉异常」是**有意的**：清理旧图失败不应该阻断主流程（后面的上传还能继续）。
- 但吞掉之前必须记日志，否则问题会被永久隐藏。
- 对比 b 节点的裸 `except:`：这里至少限定了 `Exception`，且只影响清理这一步，属于可接受的降级。
- `remove_objects` 返回的是**迭代器**，只有真正迭代时才会执行删除，所以 `for error in errors` 这层循环不能省。

**易错点**：先删旧图再上传新图，如果上传中途失败，旧图已经没了。更稳的顺序是「上传成功后再清理多余的旧对象」，或按对象名覆盖上传（同名对象直接覆盖，根本不需要先删）。

## 3.12 类型注解：容器与可调用

```python
# c_node_md_img.py:176-181
def _apply_api_rate_limit(
        self,
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60
) -> None:
```

**要点**：

- `Deque[float]` 表示「元素是 float 的双端队列」；`Deque` 来自 `typing`，`deque` 来自 `collections`，两者不是一回事（3.9+ 可直接写 `deque[float]`）。
- `-> None` 表示函数没有返回值（只产生副作用）。
- 默认参数 `window_seconds: int = 60`：调用时不传就用 60。**默认值在函数定义时求值一次**，所以不要用可变对象（list、dict）做默认值。
- 注解不影响运行，但能让 IDE 在你传错类型时给出提示。

## 3.13 本章问题清单

| 位置 | 现状 | 风险 | 建议 |
| --- | --- | --- | --- |
| 31 行 | 从 ast 导入类型名 | 被覆盖，误导读者 | 删除 |
| 36-38 行 | 三个 Django 导入未使用 | 无 Django 环境即 ImportError | 删除 |
| 104 行 | `state["md_content"]` | 缺键时 KeyError | 用 get 或先判断 |
| 121 行 | `os.listdir(images_dir)` | 目录不存在直接崩 | 先判断 is_dir |
| 147、304 行 | 循环内重复 compile | 性能浪费 | 提出循环外 |
| 172 行 | 限流 10 次与配置 15 不一致 | 行为与配置脱节 | 读配置 |
| 219 行 | 创建未使用的 ChatDeepSeek | 浪费连接与内存 | 删除 |
| 244 行 | 固定 image/jpeg | PNG 标注错误 | 按后缀推断 |
| 318 行 | URL 硬编码 http | https 部署时链接错误 | 用配置拼 |
| 353 行 | Windows 路径未加 r | 转义告警、路径异常 | 加 r 前缀 |

# 第 4 章 d 节点：引用、闭包与文本切分

`d_node_document_split.py` 干的是「把一篇 Markdown 切成若干 chunk」：先按标题粗切，没有标题就整篇兜底，再把过长的块按分隔符递归切小、把过短的相邻块合并。它是全篇算法味最重的一个文件，涉及**可变对象、闭包、状态机、正则、字符串规范化、JSON 序列化**。

## 4.1 变量、对象与引用

**核心模型**：Python 的变量不是「盒子」，而是「贴在对象上的标签」。`a = b` 不会复制对象，只是让两个名字指向同一个对象。

```python
# d_node_document_split.py:112-119
sections.append(
    {
        "title": current_title,
        "content": "\n".join(current_lines),
        "parent_title": "",
        "file_title": file_title
    }
)
```

**要点**：

- `sections.append({...})` 每次新建一个字典对象再放进列表；字典是可变对象，后续 `sec["parent_title"] = ...` 会原地修改它。
- `list.append()` 返回 `None`，不是新列表。写 `x = lst.append(1)` 会让 `x` 变成 None，是新手高频错误。
- 字符串不可变：`content.replace(...)` 返回新串，必须重新赋值。
- `"\n".join(current_lines)` 把字符串列表拼成一整段，是 `split` 的逆操作。

## 4.2 list 与 dict 常用操作

| 操作 | 列表 | 字典 |
| --- | --- | --- |
| 取元素 | `lst[i]`、`lst[-1]` | `d[key]`（缺失 KeyError） |
| 安全取 | `lst[i] if i < len(lst) else None` | `d.get(key, default)` |
| 追加 | `append(x)` 加一个；`extend(iterable)` 展开加入 | `d[k] = v` |
| 删除 | `pop()` 末尾、`pop(i)` 指定、`remove(v)` 按值 | `del d[k]`、`pop(k, default)` |
| 长度 | `len(lst)` | `len(d)` |
| 成员 | `x in lst`（按值，O(n)） | `k in d`（按键，O(1)） |
| 遍历 | `for x in lst`、`enumerate(lst)` | `keys()`、`values()`、`items()` |
| 切片 | `lst[a:b]` 返回新列表 | 不支持 |

```python
# d_node_document_split.py:159-164
refind_split = []
for sec in sections:
    refind_split.extend(self.split_long_section(sec))

final_sections = self.merge_short_section(refind_split)
```

**要点**：`split_long_section` 返回的是「块的列表」，所以用 `extend` 摊平成一维；如果用 `append`，会得到「列表的列表」。

## 4.3 嵌套函数与闭包

```python
# d_node_document_split.py:107-119
def _flush_section():
    if not current_lines:
        return
    sections.append(
        {
            "title": current_title,
            "content": "\n".join(current_lines),
            "parent_title": "",
            "file_title": file_title
        }
    )
```

**概念**：在函数内部定义的函数叫嵌套函数。如果内层函数引用了外层函数的变量（`current_lines`、`current_title`、`sections`、`file_title`），就形成**闭包**。

**要点**：

- 闭包捕获的是**变量本身**，不是定义时的值。`current_lines` 在循环里被反复重新赋值，但 `_flush_section` 每次被调用时读到的都是「当时最新的绑定」，所以结果正确。这叫「晚绑定（late binding）」——用好了很方便，用错了（比如在循环里创建 lambda）会全部拿到最后一个值。
- 内层函数要**重新赋值**外层变量，必须声明 `nonlocal`；只是调用 `append` 修改列表内容，则不需要。
- 这里用嵌套函数的好处：`_flush_section` 只在当前方法内使用，放外面会污染类命名空间；同时它直接读写局部变量，省去传参。

**易错点**：如果把 `_flush_section` 存进列表、循环结束后再统一调用，所有闭包都会看到最后一次的 `current_lines`，结果全错。这是 Python 面试高频题。

## 4.4 状态机与逐行扫描

```python
# d_node_document_split.py:99-142（节选）
lines = content.split("\n")
current_lines = []
in_code_block = False
current_title = ""

title_pattern = r"\s*#{1,6}\s+.+"

for line in lines:
    striped_line = line.strip()

    if striped_line.startswith("```") or striped_line.startswith("~~~"):
        in_code_block = not in_code_block
        current_lines.append(line)
        continue

    if (not in_code_block) and (re.match(title_pattern, line)):
        _flush_section()
        current_title = striped_line
        current_lines = [current_title]
        title_count += 1
    else:
        current_lines.append(line)

_flush_section()
```

**算法**：维护一个布尔开关 `in_code_block`，遇到代码围栏就翻转，围栏内的 `#` 一律不当标题。这是典型的**状态机**：行为取决于当前状态，状态由输入事件驱动。

**要点**：

- `in_code_block = not in_code_block` 是布尔取反，比 `if True: False else: True` 简洁。
- `continue` 跳过后面的标题判断，保证围栏行只被处理一次。
- `re.match(pattern, line)` 从字符串**开头**尝试匹配，正好对应「标题必须在行首（可有前导空白）」。
- 模式 `\s*#{1,6}\s+.+`：`\s*` 允许缩进，`#{1,6}` 匹配 1 到 6 个 `#`，`\s+` 要求 `#` 后有空白，`.+` 要求至少一个字符。所以 `## 概述` 命中，`##概述` 不命中，`####### x` 也不命中。
- 每行都 `re.match(字符串)` 会重复编译正则；数据量大时应在循环外 `pattern = re.compile(...)`，再用 `pattern.match(line)`。

**易错点**：代码围栏也可能是「三个反引号 + 语言名」的形式，用 startswith 判断前三个字符即可覆盖；正文里出现单个反引号不会被误判。另外，未闭合的围栏会让后面全部内容被当成代码，需要容错。

## 4.5 字符串规范化：统一换行符

```python
# d_node_document_split.py:84
content = content.replace("\r\n", "\n").replace("\r", "\n")
```

**要点**：

- Windows 换行是 `\r\n`，老式 Mac 是 `\r`，Unix 是 `\n`。统一成 `\n` 后，后面的 `split("\n")`、按行计数、正则 `$` 才可靠。
- **顺序不能反**：先替换 `\r\n` 再替换 `\r`。如果先替换 `\r`，`\r\n` 会变成 `\n\n`，凭空多出空行。
- 也可以 `content.splitlines()` 一次性处理所有换行风格，再 `"\n".join(...)`，更省事。

## 4.6 累加器模式：合并短块

```python
# d_node_document_split.py:251-287（节选）
merged_sections = []
current_chunk = None

for sec in sections:
    if current_chunk is None:
        current_chunk = sec
        continue

    is_current_short = len(current_chunk["content"]) < self.config.min_content_length
    is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

    if is_current_short and is_same_parent:
        parent_title = sec.get("parent_title", "")
        next_content = sec["content"]
        if parent_title and next_content.startswith(parent_title):
            next_content = next_content[len(parent_title):].lstrip()
        current_chunk["content"] += "\n\n" + next_content
        if "part" in sec:
            current_chunk["part"] = sec["part"]
    else:
        merged_sections.append(current_chunk)
        current_chunk = sec

if current_chunk is not None:
    merged_sections.append(current_chunk)
```

**模式**：用 `current_chunk` 做「当前正在累积的块」。能合就并进去，不能合就把它落盘到结果列表，再换新的。循环结束后别忘了最后一块——这是累加器模式最常见的漏写点。

**要点**：

- `current_chunk is None` 作为哨兵值，比用下标判断更清晰；判断 None 用 `is`。
- `current_chunk["content"] += ...` 修改的是字典里的值，字典本身是同一个对象，所以 `merged_sections` 里的引用也同步更新。
- `if "part" in sec` 是成员判断，避免 `KeyError`；短块可能来自「无标题兜底」，没有 `part` 字段。
- 合并前去掉重复的父标题（`next_content.startswith(parent_title)`），避免内容冗余，这是数据清洗的细节。

**易错点**：`current_chunk` 直接引用了输入列表里的字典，合并会**修改原对象**。如果调用方还要用原 `sections`，应先 `copy()`。d 节点在 `_step_4` 里已经生成了新列表，所以问题不大，但这是引用语义的典型坑。

## 4.7 三元表达式与短路求值

```python
# d_node_document_split.py:168-169
for sec in final_sections:
    if not sec.get("parent_title"):
        sec["parent_title"] = sec.get("title") or ""
```

**要点**：

- `or` 返回第一个为真的操作数：`sec.get("title") or ""` 表示「title 为空就用空串」，比 `if ... else ...` 简洁。
- 三元表达式：`f"{title}-{index}" if title else f"chunk - {index}"`（第 235 行），先写「真值时的结果」，再写条件，最后写 else。
- `sec.get("parent_title")` 对缺键返回 None，`not None` 为真，于是补默认值。这里用 `get` 而不是 `[]`，是因为「无标题兜底」生成的字典没有 `parent_title` 键。
- 注意 `or` 会把 `0`、`""`、`[]` 都当成假，若合法值是 0 要改用 `if x is None`。

## 4.8 递归字符切分器

```python
# d_node_document_split.py:209-222
prefix = f"{title}\n\n" if title else ""
available_len = get_config().max_content_length - len(prefix)

body = content
if title and body.lstrip().startswith(title):
    body = body[body.find(title) + len(title):].lstrip()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=available_len,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
)
```

**概念**：`RecursiveCharacterTextSplitter` 是 LangChain 的文本切分器。它按 `separators` 的顺序尝试切分：先按空行（段落）切，段落还太长就按换行切，再不行按中文句号、感叹号……直到块小于 `chunk_size`。

**要点**：

- `chunk_size` 是「每块最大字符数」，`chunk_overlap` 是相邻块重叠字符数（这里 0，即不重叠）。
- 之所以要 `available_len = max - len(prefix)`：每块最终会拼上 `标题\n\n`，要把标题占用的长度扣掉，保证拼完不超限。
- `separators` 里中文标点和英文标点都列了，因为文档可能中英混排；最后一个 `" "` 是兜底，再切不动就只能按空格。
- `body[body.find(title) + len(title):]` 先定位标题再截断，避免正文里重复出现标题。`body.find` 找不到返回 -1，逻辑上已由 `startswith` 保证能找到。

## 4.9 JSON 序列化：dump 与 dumps

```python
# d_node_document_split.py:186-194
path = Path(state.get("md_path")).parent / f'{state.get("file_title")}_chunks.json'
with open(path, "w", encoding="utf-8") as f:
    json.dump(
        sections,
        f,
        ensure_ascii=False,
        indent=2
    )
```

**要点**：

- `json.dump(obj, fp)` 把对象写进**文件对象**；`json.dumps(obj)` 返回**字符串**。参数顺序是「先对象，后文件」，容易写反。
- `ensure_ascii=False` 让中文原样输出，否则会变成 `\u4e2d\u6587`，文件可读性差。
- `indent=2` 缩进 2 空格，便于人眼检查；生产环境的中间备份可以不加缩进以减小体积。
- `encoding="utf-8"` 必须显式写，否则 Windows 默认 GBK 写中文可能报错。
- JSON 只支持 dict、list、str、int、float、bool、None；`Path`、`datetime`、自定义对象都不能直接序列化，需要先转成字符串。

**易错点**：`Path(state.get("md_path"))` 在 `md_path` 为 None 时抛 `TypeError`。这里假设上游一定写了 `md_path`，属于「隐式契约」，健壮写法是先判断。

## 4.10 本章问题清单

| 位置 | 现状 | 风险 | 建议 |
| --- | --- | --- | --- |
| 8 行 | `from dataclasses import replace` 未使用 | 无用导入 | 删除 |
| 10 行 | `from sys import prefix, path` 未使用 | 与后文变量同名，遮蔽 | 删除 |
| 15 行 | `import sections` 未使用 | 模块不存在即 ImportError | 删除 |
| 123 行 | 每行 strip 两次、变量名 striped | 低效、拼写错 | 复用变量 |
| 143 行 | 打印整个 sections | 日志爆炸 | 只打印数量 |
| 186 行 | `Path(None)` 风险 | md_path 缺失即崩 | 先校验 |
| 186 行 | 变量 `path` 遮蔽导入的 sys.path | 可读性差 | 改名 backup_path |
| 209 行 | 变量 `prefix` 遮蔽 sys.prefix | 同上 | 改名 title_prefix |
| 292 行 | 直接调用 process | 绕过基类日志与包装 | 用 node(state) |

# 第 5 章 e 节点：类型、校验与向量库建模

`e_node_item_name_recognition.py` 用大模型从「文件标题 + 前几个切片」里识别商品名，回填到 state 和每个 chunk，再把商品名向量化写入 Milvus。它把**类型注解、输入校验、提示词构造、静态方法、引用语义、向量库 CRUD** 串了起来。

## 5.1 类型注解进阶

```python
# e_node_item_name_recognition.py:90
def _step_1_get_inputs(self, state) -> Tuple[str, List[Dict]]:
# e_node_item_name_recognition.py:162-163
def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
```

**读法**：从里往外读。

- `Tuple[str, List[Dict]]`：一个二元组，第一项 str，第二项「字典列表」。
- `List[Tuple[str, str, Tuple[str, str]]]`：列表，元素是三元组，第三项又是二元组。
- `Dict[str, str]`：键和值都是字符串的字典。
- `->` 后面是返回值类型。

**要点**：

- 注解是**文档**：看到 `-> Tuple[str, List[Dict]]` 就知道返回「文件名 + 切片列表」，不必读函数体。
- 注解不参与运行；`List[Dict]` 里的 `Dict` 没写键值类型，等于 `Dict[Any, Any]`，精度有限。
- Python 3.9+ 可以写内置泛型 `tuple[str, list[dict]]`（g 节点就混用了），3.10+ 还可以用 `str | None` 代替 `Optional[str]`。

## 5.2 输入校验三段式

```python
# e_node_item_name_recognition.py:99-116
file_title = state["file_title"]
if not file_title:
    raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

chunks = state.get("chunks")
if not chunks:
    raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

if not isinstance(chunks, list):
    raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)
```

**三段式**：取 → 判空 → 判类型。

**要点**：

- `state["file_title"]` 与 `state.get("chunks")` 风格不一致：前者缺键直接 KeyError（错误信息不友好），后者返回 None 再统一抛业务异常。建议统一用 `get`。
- `if not chunks` 同时拦住 None 和 `[]`，因为空列表是假值。
- `isinstance(x, list)` 判断类型，能正确处理继承关系（子类实例也算）；`type(x) == list` 不行。
- 抛自定义异常而不是 `assert`：`assert` 在 `python -O` 下会被移除，不能用于生产校验。

**扩展示例**：若还要校验每个元素是 dict，可写 `all(isinstance(c, dict) for c in chunks)`，`all` 在遇到第一个 False 时短路返回。

## 5.3 切片 + 预算 + 截断

```python
# e_node_item_name_recognition.py:128-154（节选）
k = self.config.item_name_chunk_k
chunk_size = self.config.item_name_chunk_size

parts = []
total_chars = 0
for index, chunk in enumerate(chunks[:k], start=1):
    chunk_title = chunk.get("title", "").strip()
    chunk_content = chunk.get("content", "").strip()
    piece = f"第{index}个切片:{chunk_title}\n{chunk_content}\n"
    parts.append(piece)
    total_chars += len(piece)
    if total_chars > chunk_size:
        break

context = "\n\n".join(parts).strip()
final_context = context[:chunk_size]
```

**要点**：

- `chunks[:k]` 取前 k 个，切片越界不报错（不足 k 个就全取）。
- `enumerate(..., start=1)` 同时拿到下标和元素，`start=1` 让编号从 1 开始，符合人类习惯。
- `len(str)` 返回字符数（不是字节数）；中文一个字算 1。
- 「累计 + 超限就 break」是**预算式循环**：不必遍历完，够用就停。
- 最后 `context[:chunk_size]` 再硬截断一次，防止单个切片本身就超长。字符串切片越界不会报错。
- 两次截断看似冗余，实则防御：break 只能保证「上一轮没超」，本轮加上可能就超了。

**易错点**：`chunk.get("title", "")` 返回 None 时 `.strip()` 会 `AttributeError`，因为 key 存在但值是 None。稳妥写法是 `(chunk.get("title") or "").strip()`。

## 5.4 三引号 f-string 与提示词

```python
# e_node_item_name_recognition.py:181-191
prompt = f"""请从以下信息中识别出商品名称与型号：
        文件名：{file_title}

        正文切片（用于辅助识别）：
        {context}

        要求：
        1. 返回内容为字符串形式……
"""
```

**要点**：

- 三引号字符串可以跨行，保留换行与缩进；f 前缀让 `{}` 插值。
- 上面的缩进（8 个空格）会**原样进入提示词**，既浪费 token 也可能影响模型对格式的判断。规范做法是把提示词顶格写，或用 `textwrap.dedent()` 去掉公共缩进。
- 提示词工程的三要素：角色（SystemMessage）、任务与输入（HumanMessage）、输出约束（「只返回商品名」「无法识别返回空串」）。本段把输出约束写在 prompt 里，配合第 3 步的后处理（去空白、空值兜底），形成闭环。

## 5.5 消息对象与模型调用

```python
# e_node_item_name_recognition.py:193-198
message = [
    SystemMessage("你是一个专业的商品名称识别模型,请根据提供的信息,识别商品名称,名称最好不要超过20个字"),
    HumanMessage(content=prompt)
]
response = llm_ai.invoke(message)
item_name = getattr(response, "content", "").strip()
```

**要点**：

- `SystemMessage` 设定角色与全局约束，`HumanMessage` 放具体任务，这是 Chat 模型的标准消息协议（还有 `AIMessage`、`ToolMessage`）。
- `invoke` 是同步调用，返回 `AIMessage`，正文在 `.content`；流式调用用 `stream`，异步用 `ainvoke`。
- `getattr(obj, "content", "")`：属性存在就取，不存在返回空串，避免 `AttributeError`。这是「鸭子类型」思路——不关心对象是什么类，只关心有没有这个属性。
- 对比 `hasattr(response, "content")`：`getattr` 一步完成「取值 + 兜底」，更紧凑。

## 5.6 关键字实参与客户端构造

```python
# e_node_item_name_recognition.py:172-178
llm_ai = ChatDeepSeek(
    model=lm_config.llm_model,
    api_key=lm_config.api_key,
    base_url=lm_config.base_url,
    temperature=lm_config.llm_temperature,
    extra_body={"enable_thinking": False}
)
```

**要点**：

- 全部用**关键字实参**，参数名即文档，顺序无关，也不会因为库升级调整参数顺序而传错。
- `extra_body` 是透传给底层 API 的额外字段，`enable_thinking: False` 表示关闭推理过程，直接出结果，更快更省 token。
- 每次调用都 `new` 一个客户端，没有复用连接；c 节点用 `get_llm_client` 做了缓存，这里可以统一。
- 构造函数里的默认值在函数定义时求值一次，所以库作者不会用可变对象做默认值；自己写函数时也要遵守。

## 5.7 @staticmethod：不需要 self 的方法

```python
# e_node_item_name_recognition.py:251-263
@staticmethod
def _resolve_backup_dir(state) -> Path:
    md_path = state.get("md_path")
    if md_path:
        return Path(md_path).parent
    file_dir = state.get("file_dir")
    if file_dir:
        return Path(file_dir)
    raise StateFieldError(...)
```

**三种方法对比**：

| 类型 | 第一个参数 | 能访问实例 | 能访问类 | 调用方式 |
| --- | --- | --- | --- | --- |
| 实例方法 | `self` | 是 | 是 | `obj.m()` |
| 类方法 `@classmethod` | `cls` | 否 | 是 | `Cls.m()` / `obj.m()` |
| 静态方法 `@staticmethod` | 无 | 否 | 否 | `Cls.m()` / `obj.m()` |

**要点**：

- `_resolve_backup_dir` 只依赖入参 `state`，不需要 `self`，声明成静态方法语义最准确。
- 用 `self._resolve_backup_dir(state)` 调用也可以，Python 不会自动传 self。
- 好处：明确告诉读者「这个方法不碰实例状态」，也方便单独测试。

## 5.8 字符串清洗与兜底

```python
# e_node_item_name_recognition.py:201-209
item_name = getattr(response, "content", "").strip()
item_name = (item_name.replace(" ", "")
             .replace("\n", "")
             .replace("\t", "")
             .replace("\r", ""))
if not item_name:
    item_name = file_title
```

**要点**：

- 链式调用：每个 `replace` 返回新字符串，继续调下一个，写法紧凑。
- 只去空格不够，模型常在名称里夹换行、制表符，所以四个都去。
- 「模型没识别出来就退回文件名」是**降级策略**：宁可给一个粗略的名字，也不让流程中断。
- 更彻底的清洗可以用正则 `re.sub(r"\s+", "", item_name)`，一行搞定所有空白。

## 5.9 引用语义：chunk 是同一批对象

```python
# e_node_item_name_recognition.py:224-229
state["item_name"] = item_name
for chunk in chunks:
    chunk["item_name"] = item_name
state["chunks"] = chunks
```

**要点**：

- `chunks` 是列表，元素是字典；`for chunk in chunks` 拿到的是字典的**引用**，改 `chunk["item_name"]` 就是改原字典。
- 所以第 229 行 `state["chunks"] = chunks` 其实是把同一个列表再挂回去，不是复制。
- 如果希望「不改上游数据」，必须 `copy.deepcopy(chunks)`；浅拷贝 `list(chunks)` 只复制列表本身，里面的字典仍共享。
- 这个特性在本流程里是**有意利用**的：e 节点要在每个切片上补 `item_name`，供 f 节点向量化时拼进文本。

## 5.10 向量与 Milvus 数据建模

**先理解概念**：

- **稠密向量（dense）**：固定长度（如 1024）的浮点数组，捕获语义，适合「意思相近」的检索。
- **稀疏向量（sparse）**：`{词id: 权重}` 的字典，绝大多数维度为 0，擅长关键词精确匹配。
- 两者结合叫**混合检索（hybrid search）**，再配合重排，是现代 RAG 的常见方案。

**再看建表**：

```python
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
```

**要点**：

- `auto_id=True`：主键由 Milvus 生成，插入时不用给 `pk`。
- `enable_dynamic_field=True`：允许插入 schema 里没定义的字段，灵活但会牺牲一点严谨性。
- `VARCHAR` 必须给 `max_length`；中文字符占空间更大，长度要留余量。
- `FLOAT_VECTOR` 必须给 `dim`，且要与 embedding 模型输出维度一致（这里是 1024），否则插入报错。

**索引**：

```python
index_params.add_index(field_name="dense_vector", index_type="IVF_FLAT",
                       metric_type="COSINE", params={"nlist": 128})
index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX",
                       metric_type="IP",
                       params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"})
```

- `IVF_FLAT`：倒排文件 + 原始向量，先聚类再在最近的簇里暴力比较，`nlist` 是簇数；召回与速度的平衡参数。
- `COSINE`：余弦相似度，只看方向不看长度，适合文本语义。
- `SPARSE_INVERTED_INDEX`：稀疏倒排索引；`DAAT_MAXSCORE` 是高效的稀疏检索算法；`normalize=True` 做 L2 归一化，让内积（IP）等价于余弦；`quantization="none"` 不压缩，保精度。

**写入流程**：

```python
if not client.has_collection(collection_name):
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

safe_item_name = escape_milvus_string(item_name)
filter_expr = f'item_name=="{safe_item_name}"'
client.delete(collection_name=collection_name, filter=filter_expr)

data = {"file_title": file_title, "item_name": item_name,
        "dense_vector": dense_vector, "sparse_vector": sparse_vector}
client.insert(collection_name, [data])
client.load_collection(collection_name)
```

**要点**：

- **幂等**：先按 `item_name` 删除旧记录再插入，保证同一个商品反复导入不会产生重复数据。这不是数据库的「唯一索引」，而是「手动 upsert」。
- `escape_milvus_string` 转义引号等特殊字符，防止过滤表达式被注入或语法错误。
- `insert` 接收列表，表示批量；单条也要包成 `[data]`。
- `load_collection` 把集合加载进内存，**检索前必须 load**；插入本身不要求 load。这里放在插入后，更多是「确保后续可查」的习惯。
- **风险**：delete 与 insert 不是事务，如果 insert 失败，旧数据已经删了。生产环境应先写新记录再删旧的，或用版本号/时间戳做逻辑覆盖。

## 5.11 本章问题清单

| 位置 | 现状 | 风险 | 建议 |
| --- | --- | --- | --- |
| 80、232 行 | 两个方法都叫 _step_5 | 编号冲突，阅读混乱 | 重新编号 |
| 99 行 | `state["file_title"]` | 缺键 KeyError | 用 get + 校验 |
| 141 行 | `chunk.get("title", "").strip()` | 值为 None 时崩 | `or ""` 兜底 |
| 172 行 | 每次新建 ChatDeepSeek | 连接不复用 | 用客户端工厂缓存 |
| 309-341 行 | 集合已存在时仍构建 schema | 无效开销 | 先 has_collection 再构建 |
| 376 行 | 只在创建时建索引 | 已有集合缺索引不会被补 | 单独判断索引 |
| 396 行 | 每次 load_collection | 重复加载开销 | 按需加载 |
| 399 行 | 重复写 item_name | 冗余 | 删除 |


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
