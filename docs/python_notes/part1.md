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

