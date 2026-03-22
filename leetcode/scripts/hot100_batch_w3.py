# -*- coding: utf-8 -*-
"""Hot100 第 3 批题解覆盖。"""

OVERRIDES = {
    94: r"""# 94. 二叉树的中序遍历

**标签**：`栈` `树` `DFS` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现（迭代）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def inorderTraversal(self, root: TreeNode | None) -> list[int]:
        res, st = [], []
        cur = root
        while cur or st:
            while cur:
                st.append(cur)
                cur = cur.left
            cur = st.pop()
            res.append(cur.val)
            cur = cur.right
        return res
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(h)

---
""",
    96: r"""# 96. 不同的二叉搜索树

**标签**：`动态规划` `卡特兰数` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[n]
```

---

## 复杂度

- **时间**：O(n²)，**空间**：O(n)

---
""",
    98: r"""# 98. 验证二叉搜索树

**标签**：`树` `DFS` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

BST 中序遍历为严格递增；或每个结点值落在 `(min_v, max_v)` 内。

---

## 代码实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isValidBST(self, root: TreeNode | None) -> bool:
        def dfs(node: TreeNode | None, lo: float, hi: float) -> bool:
            if not node:
                return True
            if not (lo < node.val < hi):
                return False
            return dfs(node.left, lo, node.val) and dfs(node.right, node.val, hi)

        return dfs(root, float("-inf"), float("inf"))
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(h)

---
""",
    101: r"""# 101. 对称二叉树

**标签**：`树` `递归` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isSymmetric(self, root: TreeNode | None) -> bool:
        def ok(a: TreeNode | None, b: TreeNode | None) -> bool:
            if not a and not b:
                return True
            if not a or not b:
                return False
            return (
                a.val == b.val
                and ok(a.left, b.right)
                and ok(a.right, b.left)
            )

        return ok(root.left, root.right) if root else True
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(h)

---
""",
    102: r"""# 102. 二叉树的层序遍历

**标签**：`BFS` `树` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def levelOrder(self, root: TreeNode | None) -> list[list[int]]:
        if not root:
            return []
        q = deque([root])
        ans = []
        while q:
            level = []
            for _ in range(len(q)):
                n = q.popleft()
                level.append(n.val)
                if n.left:
                    q.append(n.left)
                if n.right:
                    q.append(n.right)
            ans.append(level)
        return ans
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(n)

---
""",
    104: r"""# 104. 二叉树的最大深度

**标签**：`树` `DFS` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxDepth(self, root: TreeNode | None) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(h)

---
""",
    105: r"""# 105. 从前序与中序遍历构造二叉树

**标签**：`树` `分治` `哈希` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode | None:
        idx = {v: i for i, v in enumerate(inorder)}

        def dfs(ps: int, pe: int, ins: int, ine: int) -> TreeNode | None:
            if ps > pe:
                return None
            root_val = preorder[ps]
            mid = idx[root_val]
            left_len = mid - ins
            root = TreeNode(root_val)
            root.left = dfs(ps + 1, ps + left_len, ins, mid - 1)
            root.right = dfs(ps + left_len + 1, pe, mid + 1, ine)
            return root

        return dfs(0, len(preorder) - 1, 0, len(inorder) - 1)
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(n)

---
""",
    108: r"""# 108. 将有序数组转换为二叉搜索树

**标签**：`树` `分治` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥 高频

---

## 代码实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def sortedArrayToBST(self, nums: list[int]) -> TreeNode | None:
        def dfs(lo: int, hi: int) -> TreeNode | None:
            if lo > hi:
                return None
            mid = (lo + hi) // 2
            root = TreeNode(nums[mid])
            root.left = dfs(lo, mid - 1)
            root.right = dfs(mid + 1, hi)
            return root

        return dfs(0, len(nums) - 1)
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(log n) 递归栈

---
""",
    121: r"""# 121. 买卖股票的最佳时机

**标签**：`数组` `贪心` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        mn = float("inf")
        ans = 0
        for p in prices:
            mn = min(mn, p)
            ans = max(ans, p - mn)
        return ans
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    122: r"""# 122. 买卖股票的最佳时机 II

**标签**：`贪心` `数组` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, len(prices)))
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    141: r"""# 141. 环形链表

**标签**：`链表` `双指针` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def hasCycle(self, head: ListNode | None) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow is fast:
                return True
        return False
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    142: r"""# 142. 环形链表 II

**标签**：`链表` `数学` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

快慢相遇后，一指针回 head，两指针同速再走，相遇点即环入口。

---

## 代码实现

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def detectCycle(self, head: ListNode | None) -> ListNode | None:
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow is fast:
                p = head
                while p is not slow:
                    p, slow = p.next, slow.next
                return p
        return None
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    160: r"""# 160. 相交链表

**标签**：`链表` `双指针` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(
        self, headA: ListNode, headB: ListNode
    ) -> ListNode | None:
        a, b = headA, headB
        while a is not b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a
```

---

## 复杂度

- **时间**：O(m+n)，**空间**：O(1)

---
""",
    226: r"""# 226. 翻转二叉树

**标签**：`树` `DFS` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def invertTree(self, root: TreeNode | None) -> TreeNode | None:
        if not root:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(
            root.left
        )
        return root
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(h)

---
""",
    234: r"""# 234. 回文链表

**标签**：`链表` `双指针` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 解题思路

找中点，反转后半段，双指针比较，再还原（可选）。

---

## 代码实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def isPalindrome(self, head: ListNode | None) -> bool:
        if not head or not head.next:
            return True
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
            if p1.val != p2.val:
                return False
            p1, p2 = p1.next, p2.next
        return True
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    236: r"""# 236. 二叉树的最近公共祖先

**标签**：`树` `递归` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def lowestCommonAncestor(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        if not root or root is p or root is q:
            return root
        L = self.lowestCommonAncestor(root.left, p, q)
        R = self.lowestCommonAncestor(root.right, p, q)
        if L and R:
            return root
        return L or R
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(h)

---
""",
    238: r"""# 238. 除自身以外数组的乘积

**标签**：`数组` `前缀和` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        n = len(nums)
        ans = [1] * n
        p = 1
        for i in range(n):
            ans[i] = p
            p *= nums[i]
        p = 1
        for i in range(n - 1, -1, -1):
            ans[i] *= p
            p *= nums[i]
        return ans
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1) 除输出

---
""",
    240: r"""# 240. 搜索二维矩阵 II

**标签**：`矩阵` `双指针` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 高频

---

## 解题思路

从**右上角**出发：大了往左，小了往下。

---

## 代码实现

```python
class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        if not matrix:
            return False
        r, c = 0, len(matrix[0]) - 1
        while r < len(matrix) and c >= 0:
            x = matrix[r][c]
            if x == target:
                return True
            if x > target:
                c -= 1
            else:
                r += 1
        return False
```

---

## 复杂度

- **时间**：O(m+n)，**空间**：O(1)

---
""",
}
