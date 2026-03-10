"""流式逐字解析 <video></video>：不相关内容单字符输出，video 整块输出。

最优性：时间单遍 O(n)；不嵌套 O(1) 空间用 depth，嵌套 O(层数) 用栈，均为标准最优。
"""

from typing import Callable, Generator, Iterator, Optional

# 测试用例：单层 video / 嵌套 video
inputs = "你好，xxx！<video src='vid://11223'></video>你好"
inputs_nested = "a<video>1<video>2</video>3</video>b"


def mock_stream(s: str = inputs) -> Generator[str, None, None]:
    """模拟流式输入：逐字符 yield。"""
    for c in s:
        yield c


# ---------- 共用：标签判定与读标签 ----------

_TAG = "v"  # 栈内占位，仅嵌套版使用


def _is_open(s: str) -> bool:
    """是否为 <video> 或 <video ...> 开标签（允许属性）。"""
    s = s.strip()
    if not s.startswith("<video"):
        return False
    # <video> 长度为 8 且 s[6]=='>'；<video ...> 则 s[6] 为空格
    return len(s) > 6 and s[6] in " >"


def _is_close(s: str) -> bool:
    """是否为 </video> 闭标签。"""
    return s.strip() == "</video>"  # 完整读到时无首尾空白，strip 容错


def _read_tag(it: Iterator[str]) -> str:
    """从 it 当前位（调用方已读完 '<'）一直读到 '>'，返回整段；流提前结束则返回已读部分。"""
    buf = ["<"]
    try:
        while True:
            c = next(it)
            buf.append(c)
            if c == ">":
                return "".join(buf)
    except StopIteration:
        return "".join(buf)


def _chars(stream) -> tuple[Iterator[str], Callable[[], Optional[str]]]:
    """返回 (原迭代器 it, get_char)。get_char 每次取下一字符，流结束时返回 None。"""
    it = iter(stream)

    def get_char():
        try:
            return next(it)
        except StopIteration:
            return None

    return it, get_char


def _flush(buf: list[str], raw: str):
    """流中途结束时，先按块 yield buf，再把 raw 按字符 yield。"""
    # 未闭合的 video：按单字符输出，与「严格匹配首尾」一致
    yield from "".join(buf)
    yield from raw


# ---------- 不嵌套：depth 计数 ----------


def parse_stream_no_nesting(stream) -> Generator[str, None, None]:
    """
    单层 <video>：用 depth 计数，O(1) 空间。
    不相关单字输出；整段 <video>...</video> 一块输出。
    """
    it, get = _chars(stream)
    depth = 0   # 0=在 video 外，1=在 video 内
    buf = []    # 当前 video 块缓冲（仅在 depth==1 时使用）
    c = get()

    while c is not None:
        # ---------- 在 video 外 ----------
        if depth == 0:
            if c != "<":
                yield c
                c = get()
                continue
            raw = _read_tag(it)
            if not raw.endswith(">"):
                yield from _flush([], raw)
                return
            if _is_open(raw):
                depth = 1
                buf = [raw]
            else:
                yield from raw
            c = get()
            continue

        # ---------- 在 video 内 ----------
        if c != "<":
            buf.append(c)
            c = get()
            continue

        raw = _read_tag(it)
        if not raw.endswith(">"):
            yield from _flush(buf, raw)
            return
        if _is_close(raw):
            depth = 0
            buf.append(raw)
            yield "".join(buf)
            buf = []
        else:
            buf.append(raw)
        c = get()

    # 未闭合的 video：按单字符输出，与「严格匹配首尾」一致
    yield from "".join(buf)


# ---------- 嵌套：栈 ----------


def parse_stream_nesting(stream) -> Generator[str, None, None]:
    """
    嵌套 <video>...</video>：用栈严格匹配首尾。
    仅在最外层 </video> 闭合时整块 yield（含内层）；不相关单字输出。
    """
    it, get = _chars(stream)
    stack = []  # 未闭合的 video 层，栈顶表示当前所在层
    buf = []    # 当前「从最外层 <video> 到当前位」的整段缓冲
    c = get()

    while c is not None:
        # ---------- 不在任何 video 内 ----------
        if not stack:
            if c != "<":
                yield c
                c = get()
                continue
            raw = _read_tag(it)
            if not raw.endswith(">"):
                yield from _flush([], raw)
                return
            if _is_open(raw):
                stack.append(_TAG)
                buf = [raw]
            else:
                yield from raw
            c = get()
            continue

        # ---------- 在某一层 video 内 ----------
        if c != "<":
            buf.append(c)
            c = get()
            continue

        raw = _read_tag(it)
        if not raw.endswith(">"):
            yield from _flush(buf, raw)
            return
        if _is_close(raw) and stack and stack[-1] == _TAG:
            stack.pop()
            buf.append(raw)
            if not stack:
                yield "".join(buf)  # 最外层闭合，整块输出
                buf = []
        else:
            if _is_open(raw):
                stack.append(_TAG)
            buf.append(raw)
        c = get()

    # 未闭合的 video：按单字符输出，与「严格匹配首尾」一致
    yield from "".join(buf)


# ---------- 测试 ----------


def _run(fn, s: str) -> list[str]:
    """对字符串 s 做流式解析，返回所有 yield 出的块列表。"""
    return list(fn(mock_stream(s)))


def _main() -> None:
    print("--- 不嵌套 ---")
    out_no = _run(parse_stream_no_nesting, inputs)
    for chunk in out_no:
        print(chunk)

    print("\n--- 嵌套 ---")
    out_nest = _run(parse_stream_nesting, inputs_nested)
    for chunk in out_nest:
        print(chunk)

    # 回归：拼回原串、video 整块位置、流中途结束的边界
    assert "".join(out_no) == inputs
    assert "".join(out_nest) == inputs_nested
    assert out_no[7] == "<video src='vid://11223'></video>"
    assert out_nest[1] == "<video>1<video>2</video>3</video>"
    mid = _run(parse_stream_no_nesting, "x<video")
    assert mid == ["x", "<", "v", "i", "d", "e", "o"]
    unclosed = _run(parse_stream_no_nesting, "a<video>")
    assert "".join(unclosed) == "a<video>" and unclosed == ["a", "<", "v", "i", "d", "e", "o", ">"]
    print("\n✓ 全部通过")


if __name__ == "__main__":
    _main()
