# -*- coding: utf-8 -*-
"""Hot100 第 5 批（收尾）：覆盖剩余含「待补全」的题解。"""

OVERRIDES = {
    155: r"""# 155. 最小栈

**标签**：`栈` `设计` `简单`  
**难度**：⭐ 简单  

---

## 思路

辅助栈同步维护当前最小值。

---

## 代码

```python
class MinStack:
    def __init__(self):
        self.st = []
        self.mn = []

    def push(self, val: int) -> None:
        self.st.append(val)
        self.mn.append(val if not self.mn else min(val, self.mn[-1]))

    def pop(self) -> None:
        self.st.pop()
        self.mn.pop()

    def top(self) -> int:
        return self.st[-1]

    def getMin(self) -> int:
        return self.mn[-1]
```

---
""",
    191: r"""# 191. 位 1 的个数

**标签**：`位运算` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        c = 0
        while n:
            n &= n - 1
            c += 1
        return c
```

---
""",
    213: r"""# 213. 打家劫舍 II

**标签**：`动态规划` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

环形 = 要么不抢首，要么不抢尾，两次线性打家劫舍取 max。

---

## 代码

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        def linear(arr: list[int]) -> int:
            p, q = 0, 0
            for x in arr:
                p, q = q, max(q, p + x)
            return q

        return max(linear(nums[:-1]), linear(nums[1:]))
```

---
""",
    215: r"""# 215. 数组中的第 K 大元素

**标签**：`堆` `快速选择` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码（快速选择）

```python
import random


class Solution:
    def findKthLargest(self, nums: list[int], k: int) -> int:
        k = len(nums) - k

        def part(lo: int, hi: int) -> int:
            p = random.randint(lo, hi)
            nums[p], nums[hi] = nums[hi], nums[p]
            i = lo
            for j in range(lo, hi):
                if nums[j] <= nums[hi]:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
            nums[i], nums[hi] = nums[hi], nums[i]
            return i

        lo, hi = 0, len(nums) - 1
        while True:
            m = part(lo, hi)
            if m == k:
                return nums[m]
            if m < k:
                lo = m + 1
            else:
                hi = m - 1
```

---
""",
    225: r"""# 225. 用队列实现栈

**标签**：`栈` `设计` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
from collections import deque


class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x: int) -> None:
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return len(self.q) == 0
```

---
""",
    230: r"""# 230. 二叉搜索树中第 K 小的元素

**标签**：`树` `中序` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def kthSmallest(self, root: TreeNode | None, k: int) -> int:
        st = []
        cur = root
        while cur or st:
            while cur:
                st.append(cur)
                cur = cur.left
            cur = st.pop()
            k -= 1
            if k == 0:
                return cur.val
            cur = cur.right
        return 0
```

---
""",
    232: r"""# 232. 用栈实现队列

**标签**：`栈` `设计` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class MyQueue:
    def __init__(self):
        self.in_st = []
        self.out_st = []

    def push(self, x: int) -> None:
        self.in_st.append(x)

    def pop(self) -> int:
        self._move()
        return self.out_st.pop()

    def peek(self) -> int:
        self._move()
        return self.out_st[-1]

    def empty(self) -> bool:
        return not self.in_st and not self.out_st

    def _move(self) -> None:
        if not self.out_st:
            while self.in_st:
                self.out_st.append(self.in_st.pop())
```

---
""",
    297: r"""# 297. 二叉树的序列化与反序列化

**标签**：`树` `设计` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 思路

前序 + 空结点标记（如 `None`）；反序列化用队列重建。

---

## 代码

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec:
    def serialize(self, root):
        def dfs(n):
            if not n:
                return ["#"]
            return [str(n.val)] + dfs(n.left) + dfs(n.right)

        return ",".join(dfs(root))

    def deserialize(self, data: str):
        it = iter(data.split(","))

        def dfs():
            t = next(it)
            if t == "#":
                return None
            n = TreeNode(int(t))
            n.left = dfs()
            n.right = dfs()
            return n

        return dfs()
```

---
""",
    300: r"""# 300. 最长递增子序列

**标签**：`动态规划` `二分` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码（耐心排序 / tails 二分）

```python
import bisect


class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        tails = []
        for x in nums:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)
```

---
""",
    309: r"""# 309. 最佳买卖股票时机含冷冻期

**标签**：`动态规划` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        hold, sold, rest = float("-inf"), 0, 0
        for p in prices:
            hold, sold, rest = (
                max(hold, rest - p),
                hold + p,
                max(rest, sold),
            )
        return max(sold, rest)
```

---
""",
    322: r"""# 322. 零钱兑换

**标签**：`动态规划` `完全背包` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        dp = [0] + [10**9] * amount
        for x in range(1, amount + 1):
            for c in coins:
                if c <= x:
                    dp[x] = min(dp[x], dp[x - c] + 1)
        return dp[amount] if dp[amount] < 10**9 else -1
```

---
""",
    337: r"""# 337. 打家劫舍 III

**标签**：`树形 DP` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def rob(self, root: TreeNode | None) -> int:
        def dfs(node: TreeNode | None) -> tuple[int, int]:
            if not node:
                return 0, 0
            L0, L1 = dfs(node.left)
            R0, R1 = dfs(node.right)
            return max(L0, L1) + max(R0, R1), node.val + L0 + R0

        return max(dfs(root))
```

---
""",
    338: r"""# 338. 比特位计数

**标签**：`动态规划` `位运算` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class Solution:
    def countBits(self, n: int) -> list[int]:
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)
        return dp
```

---
""",
    347: r"""# 347. 前 K 个高频元素

**标签**：`堆` `哈希` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
import heapq
from collections import Counter


class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        c = Counter(nums)
        return heapq.nlargest(k, c.keys(), key=c.__getitem__)
```

---
""",
    350: r"""# 350. 两个数组的交集 II

**标签**：`哈希` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
from collections import Counter


class Solution:
    def intersect(self, nums1: list[int], nums2: list[int]) -> list[int]:
        c1, c2 = Counter(nums1), Counter(nums2)
        ans = []
        for x in c1:
            if x in c2:
                ans.extend([x] * min(c1[x], c2[x]))
        return ans
```

---
""",
    380: r"""# 380. O(1) 时间插入、删除和获取随机元素

**标签**：`哈希` `数组` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
import random


class RandomizedSet:
    def __init__(self):
        self.a = []
        self.pos = {}

    def insert(self, val: int) -> bool:
        if val in self.pos:
            return False
        self.pos[val] = len(self.a)
        self.a.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.pos:
            return False
        i = self.pos.pop(val)
        last = self.a[-1]
        self.a[i] = last
        self.pos[last] = i
        self.a.pop()
        return True

    def getRandom(self) -> int:
        return random.choice(self.a)
```

---
""",
    381: r"""# 381. O(1) 时间插入、删除和获取随机元素 - 允许重复

**标签**：`设计` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 思路

`list` 存值，`dict` 存 `值 -> set(下标)`；删除时交换尾部与待删位置。

---

## 代码

```python
import random
from collections import defaultdict


class RandomizedCollection:
    def __init__(self):
        self.a = []
        self.idx = defaultdict(set)

    def insert(self, val: int) -> bool:
        self.idx[val].add(len(self.a))
        self.a.append(val)
        return len(self.idx[val]) == 1

    def remove(self, val: int) -> bool:
        if not self.idx[val]:
            return False
        i = self.idx[val].pop()
        last = self.a[-1]
        self.a[i] = last
        self.idx[last].discard(len(self.a) - 1)
        self.idx[last].add(i)
        self.a.pop()
        if not self.idx[val]:
            del self.idx[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.a)
```

---
""",
    394: r"""# 394. 字符串解码

**标签**：`栈` `字符串` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def decodeString(self, s: str) -> str:
        st_str, st_num = [""], [0]
        num = 0
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == "[":
                st_str.append("")
                st_num.append(num)
                num = 0
            elif c == "]":
                pat = st_str.pop()
                k = st_num.pop()
                st_str[-1] += pat * k
            else:
                st_str[-1] += c
        return st_str[0]
```

---
""",
    416: r"""# 416. 分割等和子集

**标签**：`0-1 背包` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        s = sum(nums)
        if s % 2:
            return False
        t = s // 2
        dp = [False] * (t + 1)
        dp[0] = True
        for x in nums:
            for j in range(t, x - 1, -1):
                dp[j] = dp[j] or dp[j - x]
        return dp[t]
```

---
""",
    417: r"""# 417. 太平洋大西洋水流问题

**标签**：`DFS` `矩阵` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

从太平洋边界、大西洋边界分别 DFS/BFS，能到达的格子做标记；交集为答案。

---

## 代码

```python
class Solution:
    def pacificAtlantic(self, heights: list[list[int]]) -> list[list[int]]:
        if not heights:
            return []
        m, n = len(heights), len(heights[0])
        pac = [[False] * n for _ in range(m)]
        atl = [[False] * n for _ in range(m)]

        def dfs(i: int, j: int, oc: list[list[bool]], pre: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or oc[i][j] or heights[i][j] < pre:
                return
            oc[i][j] = True
            h = heights[i][j]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dfs(i + di, j + dj, oc, h)

        for j in range(n):
            dfs(0, j, pac, heights[0][j])
            dfs(m - 1, j, atl, heights[m - 1][j])
        for i in range(m):
            dfs(i, 0, pac, heights[i][0])
            dfs(i, n - 1, atl, heights[i][n - 1])
        return [[i, j] for i in range(m) for j in range(n) if pac[i][j] and atl[i][j]]
```

---
""",
    438: r"""# 438. 找到字符串中所有字母异位词

**标签**：`滑动窗口` `哈希` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
from collections import Counter


class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        need = Counter(p)
        win = Counter()
        valid = 0
        req = len(need)
        ans = []
        left = 0
        for right, c in enumerate(s):
            win[c] += 1
            if c in need and win[c] == need[c]:
                valid += 1
            while right - left + 1 >= len(p):
                if valid == req:
                    ans.append(left)
                d = s[left]
                win[d] -= 1
                if d in need and win[d] < need[d]:
                    valid -= 1
                left += 1
        return ans
```

---
""",
    450: r"""# 450. 删除二叉搜索树中的节点

**标签**：`BST` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def deleteNode(self, root: TreeNode | None, key: int) -> TreeNode | None:
        if not root:
            return None
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            succ = root.right
            while succ.left:
                succ = succ.left
            root.val = succ.val
            root.right = self.deleteNode(root.right, succ.val)
        return root
```

---
""",
    455: r"""# 455. 分发饼干

**标签**：`贪心` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class Solution:
    def findContentChildren(self, g: list[int], s: list[int]) -> int:
        g.sort()
        s.sort()
        i = 0
        for x in s:
            if i < len(g) and x >= g[i]:
                i += 1
        return i
```

---
""",
    461: r"""# 461. 汉明距离

**标签**：`位运算` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        z = x ^ y
        c = 0
        while z:
            z &= z - 1
            c += 1
        return c
```

---
""",
    494: r"""# 494. 目标和

**标签**：`动态规划` `背包` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

设正子集和为 P，总和 S，目标 t：2P = S + t → P = (S+t)/2，转化为 0/1 背包计数。

---

## 代码

```python
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        s = sum(nums)
        if (s + target) % 2 or s < abs(target):
            return 0
        t = (s + target) // 2
        dp = [0] * (t + 1)
        dp[0] = 1
        for x in nums:
            for j in range(t, x - 1, -1):
                dp[j] += dp[j - x]
        return dp[t]
```

---
""",
    538: r"""# 538. 把二叉搜索树转换为累加树

**标签**：`BST` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def convertBST(self, root: TreeNode | None) -> TreeNode | None:
        s = 0

        def dfs(n: TreeNode | None) -> None:
            nonlocal s
            if not n:
                return
            dfs(n.right)
            s += n.val
            n.val = s
            dfs(n.left)

        dfs(root)
        return root
```

---
""",
    542: r"""# 542. 0 和 1 的矩阵

**标签**：`BFS` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

多源 BFS：所有 `0` 距离为 0 入队，`1` 初值为无穷；逐层更新四邻。

---

## 代码

```python
from collections import deque


class Solution:
    def updateMatrix(self, mat: list[list[int]]) -> list[list[int]]:
        m, n = len(mat), len(mat[0])
        dist = [[0] * n for _ in range(m)]
        q = deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    q.append((i, j))
                else:
                    dist[i][j] = 10**9
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while q:
            i, j = q.popleft()
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and dist[ni][nj] > dist[i][j] + 1:
                    dist[ni][nj] = dist[i][j] + 1
                    q.append((ni, nj))
        return dist
```

---
""",
    543: r"""# 543. 二叉树的直径

**标签**：`树` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def diameterOfBinaryTree(self, root: TreeNode | None) -> int:
        self.ans = 0

        def dep(n: TreeNode | None) -> int:
            if not n:
                return 0
            L, R = dep(n.left), dep(n.right)
            self.ans = max(self.ans, L + R)
            return max(L, R) + 1

        dep(root)
        return self.ans
```

---
""",
    560: r"""# 560. 和为 K 的子数组

**标签**：`前缀和` `哈希` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
from collections import defaultdict


class Solution:
    def subarraySum(self, nums: list[int], k: int) -> int:
        cnt = defaultdict(int)
        cnt[0] = 1
        s = ans = 0
        for x in nums:
            s += x
            ans += cnt[s - k]
            cnt[s] += 1
        return ans
```

---
""",
    572: r"""# 572. 另一棵树的子树

**标签**：`树` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isSubtree(self, root: TreeNode | None, subRoot: TreeNode | None) -> bool:
        def same(a, b):
            if not a and not b:
                return True
            if not a or not b:
                return False
            return a.val == b.val and same(a.left, b.left) and same(a.right, b.right)

        def dfs(n):
            if not n:
                return False
            return same(n, subRoot) or dfs(n.left) or dfs(n.right)

        return dfs(root)
```

---
""",
    617: r"""# 617. 合并二叉树

**标签**：`树` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def mergeTrees(
        self, root1: TreeNode | None, root2: TreeNode | None
    ) -> TreeNode | None:
        if not root1:
            return root2
        if not root2:
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1
```

---
""",
    621: r"""# 621. 任务调度器

**标签**：`贪心` `中等`  
**难度**：⭐⭐ 中等  

---

## 思路

设最高频任务数 `maxc`，其个数 `nmax`；答案至少 `(maxc-1)*(n+1)+nmax`。

---

## 代码

```python
from collections import Counter


class Solution:
    def leastInterval(self, tasks: list[str], n: int) -> int:
        c = Counter(tasks)
        mx = max(c.values())
        nmax = sum(1 for v in c.values() if v == mx)
        return max(len(tasks), (mx - 1) * (n + 1) + nmax)
```

---
""",
    647: r"""# 647. 回文子串

**标签**：`中心扩展` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0

        def expand(l: int, r: int) -> None:
            nonlocal ans
            while l >= 0 and r < n and s[l] == s[r]:
                ans += 1
                l -= 1
                r += 1

        for i in range(n):
            expand(i, i)
            expand(i, i + 1)
        return ans
```

---
""",
    704: r"""# 704. 二分查找

**标签**：`二分` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1
```

---
""",
    707: r"""# 707. 设计链表

**标签**：`设计` `链表` `中等`  
**难度**：⭐⭐ 中等  

---

## 说明

实现 `get/addAtHead/addAtTail/addAtIndex/deleteAtIndex`；用**虚拟头节点**简化。

---

## 代码（节选）

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MyLinkedList:
    def __init__(self):
        self.dummy = ListNode()
        self.size = 0

    def _get(self, index: int) -> ListNode | None:
        if index < 0:
            return None
        cur = self.dummy
        for _ in range(index + 1):
            if not cur:
                return None
            cur = cur.next
        return cur

    def get(self, index: int) -> int:
        n = self._get(index)
        return n.val if n else -1

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return
        index = max(0, index)
        pre = self._get(index - 1)
        if not pre:
            return
        pre.next = ListNode(val, pre.next)
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        pre = self._get(index - 1)
        if pre and pre.next:
            pre.next = pre.next.next
            self.size -= 1
```

---
""",
    208: r"""# 208. 实现 Trie (前缀树)

**标签**：`Trie` `设计` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        n = self.root
        for c in word:
            if c not in n:
                n[c] = {}
            n = n[c]
        n["#"] = True

    def search(self, word: str) -> bool:
        n = self.root
        for c in word:
            if c not in n:
                return False
            n = n[c]
        return "#" in n

    def startsWith(self, prefix: str) -> bool:
        n = self.root
        for c in prefix:
            if c not in n:
                return False
            n = n[c]
        return True
```

---
""",
    451: r"""# 451. 根据字符出现频率排序

**标签**：`哈希` `排序` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
from collections import Counter


class Solution:
    def frequencySort(self, s: str) -> str:
        c = Counter(s)
        return "".join(k * v for k, v in sorted(c.items(), key=lambda x: -x[1]))
```

---
""",
    733: r"""# 733. 图像渲染

**标签**：`DFS` `BFS` `简单`  
**难度**：⭐ 简单  

---

## 代码

```python
class Solution:
    def floodFill(
        self, image: list[list[int]], sr: int, sc: int, color: int
    ) -> list[list[int]]:
        old = image[sr][sc]
        if old == color:
            return image
        m, n = len(image), len(image[0])

        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or image[i][j] != old:
                return
            image[i][j] = color
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dfs(i + di, j + dj)

        dfs(sr, sc)
        return image
```

---
""",
    714: r"""# 714. 买卖股票的最佳时机含手续费

**标签**：`动态规划` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
class Solution:
    def maxProfit(self, prices: list[int], fee: int) -> int:
        hold, cash = float("-inf"), 0
        for p in prices:
            hold, cash = max(hold, cash - p), max(cash, hold + p - fee)
        return cash
```

---
""",
    912: r"""# 912. 排序数组

**标签**：`排序` `中等`  
**难度**：⭐⭐ 中等  

---

## 说明

力扣要求实现排序；面试可写**归并排序**或**快排**。

---

## 代码（归并）

```python
class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        L = self.sortArray(nums[:mid])
        R = self.sortArray(nums[mid:])
        i = j = 0
        res = []
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                res.append(L[i])
                i += 1
            else:
                res.append(R[j])
                j += 1
        res.extend(L[i:])
        res.extend(R[j:])
        return res
```

---
""",
    994: r"""# 994. 腐烂的橘子

**标签**：`BFS` `中等`  
**难度**：⭐⭐ 中等  

---

## 代码

```python
from collections import deque


class Solution:
    def orangesRotting(self, grid: list[list[int]]) -> int:
        m, n = len(grid), len(grid[0])
        q = deque()
        fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    q.append((i, j, 0))
                elif grid[i][j] == 1:
                    fresh += 1
        if fresh == 0:
            return 0
        t = 0
        while q:
            i, j, t = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    grid[ni][nj] = 2
                    fresh -= 1
                    q.append((ni, nj, t + 1))
        return t if fresh == 0 else -1
```

---
""",
    212: r"""# 212. 单词搜索 II

**标签**：`Trie` `回溯` `困难`  
**难度**：⭐⭐⭐ 困难  

---

## 说明

将 `words` 建 **Trie**，网格 DFS，走到结点带 `#` 的单词则加入答案并剪枝。

---

## 代码（框架）

```python
class Solution:
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        trie = {}
        for w in words:
            n = trie
            for c in w:
                n = n.setdefault(c, {})
            n["#"] = w
        m, n = len(board), len(board[0])
        ans = []

        def dfs(i: int, j: int, node: dict) -> None:
            if "#" in node:
                ans.append(node.pop("#"))
            if i < 0 or i >= m or j < 0 or j >= n:
                return
            c = board[i][j]
            if c not in node:
                return
            board[i][j] = "#"
            nxt = node[c]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dfs(i + di, j + dj, nxt)
            board[i][j] = c
            if not nxt:
                node.pop(c)

        for i in range(m):
            for j in range(n):
                dfs(i, j, trie)
        return ans
```

---

## 注意

边界与 Trie 删除需按题解细调；以上为常见写法思路。

---
""",
}
