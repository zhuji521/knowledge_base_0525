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

