# 拼多多一面：反向定位字符（斐波那契拼接字符串）

## 题目

给定：

- `S1 = s1`
- `S2 = s2`
- `Sm = S(m-2) + S(m-1)`（`m >= 3`）

求第 `m` 个字符串中的第 `n` 个字符（下标从 1 开始）。

---

## 核心思路（不能构造大串）

字符串长度会很快爆炸，必须“只算长度 + 反向定位”。

1. 先计算长度数组：
   - `L1 = len(s1)`
   - `L2 = len(s2)`
   - `Li = L(i-2) + L(i-1)`

2. 反推位置：
   - `Sm = S(m-2) + S(m-1)`
   - 若 `n <= L(m-2)`，目标在左半，令 `m = m-2`
   - 否则在右半，令 `n -= L(m-2)`，`m = m-1`

3. 不断缩小直到 `m == 1` 或 `m == 2`，直接从 `s1/s2` 取字符。

---

## Python 实现

```python
def kth_char_fib_string(s1: str, s2: str, m: int, n: int) -> str:
    # 1) 长度预处理
    L = [0] * (m + 1)
    if m >= 1:
        L[1] = len(s1)
    if m >= 2:
        L[2] = len(s2)
    for i in range(3, m + 1):
        L[i] = L[i - 2] + L[i - 1]

    # 边界检查
    if m < 1 or n < 1 or n > L[m]:
        raise IndexError("n out of range")

    # 2) 反向定位
    while m > 2:
        if n <= L[m - 2]:
            m = m - 2
        else:
            n -= L[m - 2]
            m = m - 1

    # 3) 落到基串
    return s1[n - 1] if m == 1 else s2[n - 1]
```

---

## 复杂度

- 时间复杂度：`O(m)`
- 空间复杂度：`O(m)`（长度数组）

---

## 易错点

1. `n` 是 1-based，下标取字符时要 `n-1`。
2. 不要真的构造 `Sm`，会超时/爆内存。
3. 先检查 `n <= L[m]`，防止越界。

