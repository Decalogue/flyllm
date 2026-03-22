# -*- coding: utf-8 -*-
"""Hot100 第 4 批题解覆盖。"""

OVERRIDES = {
    114: r"""# 114. 二叉树展开为链表

**标签**：`树` `链表` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

前序遍历顺序展开：右子树暂存，左子树移到右，再接上暂存的右子树。

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def flatten(self, root: TreeNode | None) -> None:
        cur = root
        while cur:
            if cur.left:
                p = cur.left
                while p.right:
                    p = p.right
                p.right = cur.right
                cur.right = cur.left
                cur.left = None
            cur = cur.right
```

---

## 复杂度

- **时间** O(n)，**空间** O(1)

---
""",
    115: r"""# 115. 不同的子序列

**标签**：`动态规划` `字符串` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 代码

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j]
                if s[i - 1] == t[j - 1]:
                    dp[i][j] += dp[i - 1][j - 1]
        return dp[m][n]
```

---

## 复杂度

- O(mn) 时间，O(mn) 空间（可滚为一维）

---
""",
    123: r"""# 123. 买卖股票的最佳时机 III

**标签**：`动态规划` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 思路

最多两笔交易：状态 `buy1/sell1/buy2/sell2` 或 `dp[k][i]`。

---

## 代码

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        b1 = b2 = float("inf")
        s1 = s2 = 0
        for p in prices:
            b1 = min(b1, p)
            s1 = max(s1, p - b1)
            b2 = min(b2, p - s1)
            s2 = max(s2, p - b2)
        return s2
```

---

## 复杂度

- O(n) 时间，O(1) 空间

---
""",
    124: r"""# 124. 二叉树中的最大路径和

**标签**：`树` `DFS` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxPathSum(self, root: TreeNode | None) -> int:
        self.ans = float("-inf")

        def dfs(node: TreeNode | None) -> int:
            if not node:
                return 0
            L = max(0, dfs(node.left))
            R = max(0, dfs(node.right))
            self.ans = max(self.ans, node.val + L + R)
            return node.val + max(L, R)

        dfs(root)
        return self.ans
```

---

## 复杂度

- O(n) 时间，O(h) 空间

---
""",
    127: r"""# 127. 单词接龙

**标签**：`BFS` `字符串` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 思路

双向 BFS + 邻接预处理；每次只改一个字母，用队列扩展最短层数。

---

## 代码（框架）

```python
from collections import deque


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        words = set(wordList)
        if endWord not in words:
            return 0
        q = deque([(beginWord, 1)])
        seen = {beginWord}
        while q:
            w, d = q.popleft()
            if w == endWord:
                return d
            for i in range(len(w)):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    nw = w[:i] + c + w[i + 1 :]
                    if nw in words and nw not in seen:
                        seen.add(nw)
                        q.append((nw, d + 1))
        return 0
```

---

## 说明

数据规模大时需用「通用状态」或预处理邻接以优化；以上为朴素 BFS 写法，力扣部分用例可通过，极大数据需剪枝/双向 BFS。

---
""",
    130: r"""# 130. 被围绕的区域

**标签**：`DFS` `并查集` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

从**边界上**的 `O` 开始 DFS/BFS，标记与边界连通的 `O`；最后遍历，未标记的 `O` 变 `X`。

---

## 代码

```python
class Solution:
    def solve(self, board: list[list[str]]) -> None:
        if not board:
            return
        m, n = len(board), len(board[0])

        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != "O":
                return
            board[i][j] = "#"
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dfs(i + di, j + dj)

        for j in range(n):
            dfs(0, j)
            dfs(m - 1, j)
        for i in range(m):
            dfs(i, 0)
            dfs(i, n - 1)
        for i in range(m):
            for j in range(n):
                if board[i][j] == "O":
                    board[i][j] = "X"
                elif board[i][j] == "#":
                    board[i][j] = "O"
```

---

## 复杂度

- O(mn)

---
""",
    134: r"""# 134. 加油站

**标签**：`贪心` `数组` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        total = start = 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            if total < 0:
                total = 0
                start = i + 1
        return start
```

---

## 复杂度

- O(n)

---
""",
    136: r"""# 136. 只出现一次的数字

**标签**：`位运算` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class Solution:
    def singleNumber(self, nums: list[int]) -> int:
        x = 0
        for n in nums:
            x ^= n
        return x
```

---

## 复杂度

- O(n) 时间，O(1) 空间

---
""",
    138: r"""# 138. 随机链表的复制

**标签**：`哈希表` `链表` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

旧结点与新结点交错插入，再拆分；或哈希表 `old->new`。

---

## 代码（哈希表）

```python
class Node:
    def __init__(self, x: int, next: "Node | None" = None, random: "Node | None" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: "Node | None") -> "Node | None":
        if not head:
            return None
        d = {}
        cur = head
        while cur:
            d[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur:
            d[cur].next = d.get(cur.next)
            d[cur].random = d.get(cur.random)
            cur = cur.next
        return d[head]
```

---

## 复杂度

- O(n) 时间，O(n) 空间

---
""",
    139: r"""# 139. 单词拆分

**标签**：`动态规划` `字符串` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        ws = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in ws:
                    dp[i] = True
                    break
        return dp[n]
```

---

## 复杂度

- O(n²·L) 最坏，可优化

---
""",
    143: r"""# 143. 重排链表

**标签**：`链表` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

找中点、反转后半、交错合并。

---

## 代码

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reorderList(self, head: ListNode | None) -> None:
        if not head or not head.next:
            return
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        prev, cur = None, slow.next
        slow.next = None
        while cur:
            nxt = cur.next
            cur.next = prev
            prev, cur = cur, nxt
        p1, p2 = head, prev
        while p2:
            n1, n2 = p1.next, p2.next
            p1.next, p2.next = p2, n1
            p1, p2 = n1, n2
```

---

## 复杂度

- O(n)

---
""",
    144: r"""# 144. 二叉树的前序遍历

**标签**：`栈` `树` `简单`  
**难度**：⭐ 简单  

---

## 代码（迭代）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def preorderTraversal(self, root: TreeNode | None) -> list[int]:
        res, st = [], [root] if root else []
        while st:
            n = st.pop()
            res.append(n.val)
            if n.right:
                st.append(n.right)
            if n.left:
                st.append(n.left)
        return res
```

---

## 复杂度

- O(n)

---
""",
    145: r"""# 145. 二叉树的后序遍历

**标签**：`栈` `树` `简单`  
**难度**：⭐ 简单  

---

## 代码（迭代）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def postorderTraversal(self, root: TreeNode | None) -> list[int]:
        if not root:
            return []
        res, st = [], [root]
        while st:
            n = st.pop()
            res.append(n.val)
            if n.left:
                st.append(n.left)
            if n.right:
                st.append(n.right)
        return res[::-1]
```

---

## 复杂度

- O(n)

---
""",
    148: r"""# 148. 排序链表

**标签**：`链表` `归并排序` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

快慢找中点断开，递归归并两条有序链表。

---

## 代码

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def sortList(self, head: ListNode | None) -> ListNode | None:
        if not head or not head.next:
            return head

        def merge(a: ListNode | None, b: ListNode | None) -> ListNode | None:
            dummy = ListNode()
            cur = dummy
            while a and b:
                if a.val <= b.val:
                    cur.next, a = a, a.next
                else:
                    cur.next, b = b, b.next
                cur = cur.next
            cur.next = a or b
            return dummy.next

        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        mid = slow.next
        slow.next = None
        return merge(self.sortList(head), self.sortList(mid))
```

---

## 复杂度

- O(n log n) 时间，O(log n) 栈

---
""",
    152: r"""# 152. 乘积最大子数组

**标签**：`动态规划` `数组` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        mx = mn = ans = nums[0]
        for x in nums[1:]:
            t = (mx * x, mn * x, x)
            mx, mn = max(t), min(t)
            ans = max(ans, mx)
        return ans
```

---

## 复杂度

- O(n)

---
""",
    153: r"""# 153. 寻找旋转排序数组中的最小值

**标签**：`二分查找` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def findMin(self, nums: list[int]) -> int:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] > nums[hi]:
                lo = mid + 1
            else:
                hi = mid
        return nums[lo]
```

---

## 复杂度

- O(log n)

---
""",
    188: r"""# 188. 买卖股票的最佳时机 IV

**标签**：`动态规划` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 思路

通用 k 笔：`dp[t][i]` 第 i 天完成第 t 笔交易的最大收益；或压缩为 `buy/sell` 数组长度 k+1。

---

## 代码（压缩状态）

```python
class Solution:
    def maxProfit(self, k: int, prices: list[int]) -> int:
        if not prices:
            return 0
        n = len(prices)
        if k >= n // 2:
            return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, n))
        buy = [-10**9] * (k + 1)
        sell = [0] * (k + 1)
        for p in prices:
            for t in range(1, k + 1):
                buy[t] = max(buy[t], sell[t - 1] - p)
                sell[t] = max(sell[t], buy[t] + p)
        return sell[k]
```

---

## 复杂度

- O(nk)

---
""",
    198: r"""# 198. 打家劫舍

**标签**：`动态规划` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        a = b = 0
        for x in nums:
            a, b = b, max(b, a + x)
        return b
```

---

## 复杂度

- O(n) 时间，O(1) 空间

---
""",
    207: r"""# 207. 课程表

**标签**：`拓扑排序` `图` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码（Kahn）

```python
from collections import deque


class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        g = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            indeg[a] += 1
        q = deque([i for i in range(numCourses) if indeg[i] == 0])
        cnt = 0
        while q:
            u = q.popleft()
            cnt += 1
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return cnt == numCourses
```

---

## 复杂度

- O(V+E)

---
""",
    210: r"""# 210. 课程表 II

**标签**：`拓扑排序` `BFS` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
from collections import deque


class Solution:
    def findOrder(self, numCourses: int, prerequisites: list[list[int]]) -> list[int]:
        g = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            indeg[a] += 1
        q = deque([i for i in range(numCourses) if indeg[i] == 0])
        ans = []
        while q:
            u = q.popleft()
            ans.append(u)
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return ans if len(ans) == numCourses else []
```

---

## 复杂度

- O(V+E)

---
""",
}
