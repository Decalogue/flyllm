# -*- coding: utf-8 -*-
"""Hot100 第 2 批题解覆盖（合并入 OVERRIDES）。"""

OVERRIDES = {
    49: r"""# 49. 字母异位词分组

**标签**：`哈希表` `字符串` `排序` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 高频

---

## 解题思路

异位词排序后相同：用 `tuple(sorted(s))` 或 `"".join(sorted(s))` 作 key；或计数元组作 key。

---

## 代码实现

```python
from collections import defaultdict


class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        g: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for w in strs:
            g[tuple(sorted(w))].append(w)
        return list(g.values())
```

---

## 复杂度

- **时间**：O(n·k log k)，n 为串数、k 为串长；**空间**：O(nk)

---
""",
    50: r"""# 50. Pow(x, n)

**标签**：`递归` `数学` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 中频

---

## 解题思路

**快速幂**：n 为偶数时 `x^n = (x^2)^(n/2)`；为奇数时 `x * x^(n-1)`。注意 n 为负数时取倒数。

---

## 代码实现

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            x = 1 / x
            n = -n
        res = 1.0
        while n:
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res
```

---

## 复杂度

- **时间**：O(log|n|)，**空间**：O(1)

---
""",
    54: r"""# 54. 螺旋矩阵

**标签**：`矩阵` `模拟` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        if not matrix:
            return []
        res = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        while top <= bottom and left <= right:
            for c in range(left, right + 1):
                res.append(matrix[top][c])
            top += 1
            for r in range(top, bottom + 1):
                res.append(matrix[r][right])
            right -= 1
            if top <= bottom:
                for c in range(right, left - 1, -1):
                    res.append(matrix[bottom][c])
                bottom -= 1
            if left <= right:
                for r in range(bottom, top - 1, -1):
                    res.append(matrix[r][left])
                left += 1
        return res
```

---

## 复杂度

- **时间**：O(mn)，**空间**：O(1) 除输出外

---
""",
    55: r"""# 55. 跳跃游戏

**标签**：`贪心` `数组` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

维护**最远可达下标** `reach`；遍历 `i`，若 `i > reach` 失败；否则 `reach = max(reach, i+nums[i])`。

---

## 代码实现

```python
class Solution:
    def canJump(self, nums: list[int]) -> bool:
        reach = 0
        for i, x in enumerate(nums):
            if i > reach:
                return False
            reach = max(reach, i + x)
            if reach >= len(nums) - 1:
                return True
        return True
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    56: r"""# 56. 合并区间

**标签**：`排序` `数组` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        intervals.sort(key=lambda x: x[0])
        res = []
        for s, e in intervals:
            if not res or s > res[-1][1]:
                res.append([s, e])
            else:
                res[-1][1] = max(res[-1][1], e)
        return res
```

---

## 复杂度

- **时间**：O(n log n)，**空间**：O(1) 或 O(n)

---
""",
    69: r"""# 69. x 的平方根

**标签**：`二分查找` `数学` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        lo, hi = 0, x
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid * mid <= x < (mid + 1) * (mid + 1):
                return mid
            if mid * mid < x:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi
```

---

## 复杂度

- **时间**：O(log x)

---
""",
    70: r"""# 70. 爬楼梯

**标签**：`动态规划` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        a, b = 1, 2
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    72: r"""# 72. 编辑距离

**标签**：`动态规划` `字符串` `困难`  
**难度**：⭐⭐⭐ 困难  
**频率**：🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]
```

---

## 复杂度

- **时间**：O(mn)，**空间**：O(mn)（可滚为一维）

---
""",
    73: r"""# 73. 矩阵置零

**标签**：`矩阵` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 高频

---

## 解题思路

用首行首列标记该行该列是否需清零，再用两个变量记录首行/首列本身是否含 0。

---

## 代码实现

```python
class Solution:
    def setZeroes(self, matrix: list[list[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        r0 = any(matrix[0][j] == 0 for j in range(n))
        c0 = any(matrix[i][0] == 0 for i in range(m))
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if r0:
            for j in range(n):
                matrix[0][j] = 0
        if c0:
            for i in range(m):
                matrix[i][0] = 0
```

---

## 复杂度

- **时间**：O(mn)，**空间**：O(1)

---
""",
    78: r"""# 78. 子集

**标签**：`回溯` `位运算` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现（迭代）

```python
class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res = [[]]
        for x in nums:
            res += [s + [x] for s in res]
        return res
```

---

## 复杂度

- **时间**：O(n·2^n)，**空间**：O(n·2^n)

---
""",
    79: r"""# 79. 单词搜索

**标签**：`回溯` `DFS` `矩阵` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def exist(self, board: list[list[str]], word: str) -> bool:
        m, n = len(board), len(board[0])

        def dfs(i: int, j: int, k: int) -> bool:
            if k == len(word):
                return True
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
                return False
            t, board[i][j] = board[i][j], "#"
            ok = (
                dfs(i + 1, j, k + 1)
                or dfs(i - 1, j, k + 1)
                or dfs(i, j + 1, k + 1)
                or dfs(i, j - 1, k + 1)
            )
            board[i][j] = t
            return ok

        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
```

---

## 复杂度

- **时间**：O(mn·3^L)，L 为单词长度

---
""",
    91: r"""# 91. 解码方法

**标签**：`动态规划` `字符串` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == "0":
            return 0
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
            if s[i - 1] != "0":
                dp[i] += dp[i - 1]
            two = int(s[i - 2 : i])
            if 10 <= two <= 26:
                dp[i] += dp[i - 2]
        return dp[n]
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(n) 可优化为 O(1)

---
""",
}
