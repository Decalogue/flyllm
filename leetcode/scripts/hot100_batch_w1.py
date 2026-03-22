# -*- coding: utf-8 -*-
"""Hot100 缺失题解：第 1 批完整覆盖（生成器会合并到 OVERRIDES）。"""

OVERRIDES = {
    5: r"""# 5. 最长回文子串

**标签**：`字符串` `中心扩展` `动态规划` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 说明

力扣官方题号为 **5**。本仓库完整题解已写在 **`12.最长回文子串.md`**（历史命名），内容含中心扩展、DP、Manacher 与图示，请直接打开该文件阅读，避免两处重复维护。

---
""",
    3: r"""# 3. 无重复字符的最长子串

**标签**：`滑动窗口` `哈希表` `字符串` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 题目描述

给定字符串 `s`，求**不含重复字符**的**最长连续子串**的长度。

---

## 📋 面试要点速查

**一句话总结**：右端扩展窗口，用哈希记录字符最后出现位置；发现重复则左端跳到重复字符的下一位。

**关键词**：`滑动窗口` `last_index` `O(n)`

- **时间复杂度**：O(n)
- **空间复杂度**：O(min(n, |Σ|))

---

## 解题思路

用字典 `last` 记录每个字符**最后一次出现下标**。`right` 右移时，若 `s[right]` 在窗口内重复，则 `left = max(left, last[c] + 1)`，再更新答案与 `last[c]`。

---

## 代码实现

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}
        left = 0
        ans = 0
        for right, c in enumerate(s):
            if c in last and last[c] >= left:
                left = last[c] + 1
            last[c] = right
            ans = max(ans, right - left + 1)
        return ans
```

---

## 复杂度分析

- **时间**：O(n)，双指针各至多走 n 步。
- **空间**：O(min(n, |Σ|))。

---
""",
    4: r"""# 4. 寻找两个正序数组的中位数

**标签**：`二分查找` `数组` `困难`  
**难度**：⭐⭐⭐ 困难  
**频率**：🔥🔥 中频

---

## 题目描述

给定两个**正序**数组 `nums1`、`nums2`，长度分别为 `m`、`n`，求两个数组合并后的**中位数**。要求时间复杂度 **O(log(m+n))**。

---

## 解题思路

等价于在**两个有序数组**上做二分：划分 `i`（`nums1` 左半取 `i` 个）与 `j = (m+n+1)//2 - i`，使左半最大值 ≤ 右半最小值。比较边界 `nums1[i-1]`、`nums2[j-1]`、`nums1[i]`、`nums2[j]` 调整 `i`。

---

## 代码实现

```python
class Solution:
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        lo, hi = 0, m
        half = (m + n + 1) // 2
        while lo <= hi:
            i = (lo + hi) // 2
            j = half - i
            left1 = float("-inf") if i == 0 else nums1[i - 1]
            right1 = float("inf") if i == m else nums1[i]
            left2 = float("-inf") if j == 0 else nums2[j - 1]
            right2 = float("inf") if j == n else nums2[j]
            if left1 <= right2 and left2 <= right1:
                if (m + n) % 2:
                    return float(max(left1, left2))
                return (max(left1, left2) + min(right1, right2)) / 2.0
            if left1 > right2:
                hi = i - 1
            else:
                lo = i + 1
        return 0.0
```

---

## 复杂度分析

- **时间**：O(log(min(m,n)))
- **空间**：O(1)

---
""",
    11: r"""# 11. 盛最多水的容器

**标签**：`贪心` `双指针` `数组` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 题目描述

给定非负整数数组 `height`，选两条线使与 x 轴构成容器，求**最大水量**（面积）：`min(h[i],h[j]) * (j-i)`。

---

## 解题思路

双指针 `l,r` 从两端向中间移动。每次移动**较短边**一侧：因为宽度必减小，只有可能通过增高短板来增大面积。

---

## 代码实现

```python
class Solution:
    def maxArea(self, height: list[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            ans = max(ans, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return ans
```

---

## 复杂度分析

- **时间**：O(n)，**空间**：O(1)

---
""",
    19: r"""# 19. 删除链表的倒数第 N 个结点

**标签**：`链表` `双指针` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

**快慢指针**：先让快指针走 `n` 步，再快慢同速直到快指针到尾，慢指针停在待删**前一个**结点。用**虚拟头节点**统一删除头结点的情况。

---

## 代码实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        fast = slow = dummy
        for _ in range(n + 1):
            fast = fast.next
        while fast:
            fast, slow = fast.next, slow.next
        slow.next = slow.next.next
        return dummy.next
```

---

## 复杂度分析

- **时间**：O(L)，**空间**：O(1)

---
""",
    20: r"""# 20. 有效的括号

**标签**：`栈` `字符串` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 解题思路

栈中存左括号；遇右括号则与栈顶匹配，不匹配或栈空则无效；结束栈须为空。

---

## 代码实现

```python
class Solution:
    def isValid(self, s: str) -> bool:
        st = []
        p = {")": "(", "]": "[", "}": "{"}
        for c in s:
            if c in "([{":
                st.append(c)
            else:
                if not st or st[-1] != p[c]:
                    return False
                st.pop()
        return len(st) == 0
```

---

## 复杂度分析

- **时间**：O(n)，**空间**：O(n)

---
""",
    21: r"""# 21. 合并两个有序链表

**标签**：`链表` `递归` `简单`  
**难度**：⭐ 简单  
**频率**：🔥🔥🔥 高频

---

## 代码实现（迭代）

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(
        self, list1: ListNode | None, list2: ListNode | None
    ) -> ListNode | None:
        dummy = ListNode()
        cur = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        cur.next = list1 or list2
        return dummy.next
```

---

## 复杂度分析

- **时间**：O(m+n)，**空间**：O(1)

---
""",
    22: r"""# 22. 括号生成

**标签**：`回溯` `字符串` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

回溯生成 `n` 对括号：当前串中左括号数 `< n` 时可放 `(`；右括号数 `< 左括号数` 时可放 `)`。

---

## 代码实现

```python
class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        ans = []

        def dfs(s: str, op: int, cl: int) -> None:
            if len(s) == 2 * n:
                ans.append(s)
                return
            if op < n:
                dfs(s + "(", op + 1, cl)
            if cl < op:
                dfs(s + ")", op, cl + 1)

        dfs("", 0, 0)
        return ans
```

---

## 复杂度分析

- 卡特兰数级别个合法串，时间近似 O(4^n / sqrt(n))（输出规模）。

---
""",
    24: r"""# 24. 两两交换链表中的节点

**标签**：`链表` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 中频

---

## 代码实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def swapPairs(self, head: ListNode | None) -> ListNode | None:
        dummy = ListNode(0, head)
        pre = dummy
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next, b.next, a.next = b, a, b.next
            pre = a
        return dummy.next
```

---

## 复杂度

- **时间**：O(n)，**空间**：O(1)

---
""",
    33: r"""# 33. 搜索旋转排序数组

**标签**：`数组` `二分查找` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

二分 `mid`：至少一半区间仍有序。若 `target` 落在有序区间内则收缩到该侧，否则去另一侧。

---

## 代码实现

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return mid
            if nums[lo] <= nums[mid]:
                if nums[lo] <= target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            else:
                if nums[mid] < target <= nums[hi]:
                    lo = mid + 1
                else:
                    hi = mid - 1
        return -1
```

---

## 复杂度

- **时间**：O(log n)，**空间**：O(1)

---
""",
    34: r"""# 34. 在排序数组中查找元素的第一个和最后一个位置

**标签**：`二分查找` `数组` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

两次二分：第一次找 `>= target` 的最左；第二次找 `> target` 的最左再减一，即最后一个 `target`。

---

## 代码实现

```python
class Solution:
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        def bisect_left(a: list[int], x: int) -> int:
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        def bisect_right(a: list[int], x: int) -> int:
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        l = bisect_left(nums, target)
        if l == len(nums) or nums[l] != target:
            return [-1, -1]
        r = bisect_right(nums, target) - 1
        return [l, r]
```

---

## 复杂度

- **时间**：O(log n)，**空间**：O(1)

---
""",
    39: r"""# 39. 组合总和

**标签**：`回溯` `数组` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

回溯：排序后 DFS，每次从 `start` 起选数，和为 `target` 时收方案；可重复选同一元素故下一层仍从 `i` 开始。

---

## 代码实现

```python
class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates.sort()
        ans = []

        def dfs(start: int, rest: int, path: list[int]) -> None:
            if rest == 0:
                ans.append(path.copy())
                return
            for i in range(start, len(candidates)):
                x = candidates[i]
                if x > rest:
                    break
                path.append(x)
                dfs(i, rest - x, path)
                path.pop()

        dfs(0, target, [])
        return ans
```

---

## 复杂度

- 指数级（与解的数量相关）

---
""",
    40: r"""# 40. 组合总和 II

**标签**：`回溯` `数组` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 中频

---

## 解题思路

与 39 类似，但**每个数字只能用一次**，且需**去重**：排序后，同一层循环若 `candidates[i] == candidates[i-1]` 且 `i>start` 则跳过。

---

## 代码实现

```python
class Solution:
    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates.sort()
        ans: list[list[int]] = []

        def dfs(start: int, rest: int, path: list[int]) -> None:
            if rest == 0:
                ans.append(path.copy())
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                x = candidates[i]
                if x > rest:
                    break
                path.append(x)
                dfs(i + 1, rest - x, path)
                path.pop()

        dfs(0, target, [])
        return ans
```

---

## 复杂度

- 最坏指数级

---
""",
    46: r"""# 46. 全排列

**标签**：`回溯` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 代码实现

```python
class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        ans: list[list[int]] = []

        def dfs(path: list[int], used: list[bool]) -> None:
            if len(path) == len(nums):
                ans.append(path.copy())
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                used[i] = True
                path.append(nums[i])
                dfs(path, used)
                path.pop()
                used[i] = False

        dfs([], [False] * len(nums))
        return ans
```

---

## 复杂度

- **时间**：O(n·n!)，**空间**：O(n)

---
""",
    47: r"""# 47. 全排列 II

**标签**：`回溯` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥 中频

---

## 解题思路

排序后，同一层若 `nums[i]==nums[i-1]` 且 `i` 未使用上一轮同值则剪枝（常用 `used[i-1]` 判断）。

---

## 代码实现

```python
class Solution:
    def permuteUnique(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        ans: list[list[int]] = []
        used = [False] * len(nums)

        def dfs(path: list[int]) -> None:
            if len(path) == len(nums):
                ans.append(path.copy())
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                    continue
                used[i] = True
                path.append(nums[i])
                dfs(path)
                path.pop()
                used[i] = False

        dfs([])
        return ans
```

---

## 复杂度

- 与解的数量相关

---
""",
    48: r"""# 48. 旋转图像

**标签**：`矩阵` `中等`  
**难度**：⭐⭐ 中等  
**频率**：🔥🔥🔥 高频

---

## 解题思路

**先转置再沿竖中轴翻转**（或先水平翻转再转置），等价于顺时针 90°。

---

## 代码实现

```python
class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row.reverse()
```

---

## 复杂度

- **时间**：O(n²)，**空间**：O(1)

---
""",
}
