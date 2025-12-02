# 算法题机考准备

## 📋 快速复习清单

### 🔥 高频算法类型

#### 1. 数组和字符串
- [ ] 双指针（Two Pointers）
- [ ] 滑动窗口（Sliding Window）
- [ ] 前缀和（Prefix Sum）
- [ ] 哈希表（Hash Table）

#### 2. 链表
- [ ] 反转链表
- [ ] 快慢指针
- [ ] 合并链表
- [ ] 链表环检测

#### 3. 树
- [ ] 二叉树遍历（前序/中序/后序）
- [ ] 二叉搜索树（BST）
- [ ] DFS / BFS
- [ ] 树的递归

#### 4. 图
- [ ] BFS / DFS
- [ ] 拓扑排序
- [ ] 最短路径
- [ ] 并查集（Union-Find）

#### 5. 动态规划
- [ ] 背包问题
- [ ] 最长子序列
- [ ] 股票买卖
- [ ] 路径问题

#### 6. 回溯算法
- [ ] 排列/组合
- [ ] N皇后
- [ ] 数独

#### 7. 排序和搜索
- [ ] 快速排序
- [ ] 归并排序
- [ ] 二分查找

## 📚 已准备的题目

- [1. 两数之和](https://github.com/Decalogue/flyllm/blob/main/leetcode/1.两数之和.md)
- [2. 两数相加](https://github.com/Decalogue/flyllm/blob/main/leetcode/2.两数相加.md)
- [15. 三数之和](https://github.com/Decalogue/flyllm/blob/main/leetcode/15.三数之和.md)
- [23. 合并K个升序链表](https://github.com/Decalogue/flyllm/blob/main/leetcode/23.合并K个升序链表.md)
- [200. 岛屿数量](https://github.com/Decalogue/flyllm/blob/main/leetcode/200.岛屿数量.md)
- [695. 岛屿的最大面积](https://github.com/Decalogue/flyllm/blob/main/leetcode/695.岛屿的最大面积.md)

## 🎯 机考策略

### 时间管理
- **总时长**：通常90-120分钟
- **题目数量**：2-3题（中等难度为主）
- **时间分配**：
  - 审题和思考：10-15分钟/题
  - 编码：20-30分钟/题
  - 测试和调试：10-15分钟/题
  - 预留缓冲时间：10-20分钟

### 答题步骤
1. **理解题意**（3-5分钟）
   - 仔细阅读题目描述
   - 理解输入输出格式
   - 分析边界情况
   
2. **设计算法**（5-10分钟）
   - 用伪代码或流程图
   - 考虑时间和空间复杂度
   - 评估不同解法
   
3. **编码实现**（20-30分钟）
   - 先写框架，再填细节
   - 注意变量命名清晰
   - 添加关键注释
   
4. **测试调试**（10-15分钟）
   - 用示例测试
   - 检查边界条件
   - 优化代码

### 常见陷阱
- ❌ 数组越界
- ❌ 空指针异常
- ❌ 整数溢出
- ❌ 边界条件处理
- ❌ 忘记处理空输入

## 💡 编码模板

### 双指针模板
```python
def two_pointers(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        # 处理逻辑
        if condition:
            left += 1
        else:
            right -= 1
    return result
```

### 滑动窗口模板
```python
def sliding_window(s):
    left = 0
    window = {}
    for right in range(len(s)):
        # 扩展窗口
        window[s[right]] = window.get(s[right], 0) + 1
        
        # 收缩窗口
        while need_shrink:
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1
    return result
```

### DFS模板
```python
def dfs(node, visited):
    if not node or node in visited:
        return
    
    visited.add(node)
    # 处理当前节点
    
    for neighbor in node.neighbors:
        dfs(neighbor, visited)
```

### BFS模板
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = set([start])
    
    while queue:
        node = queue.popleft()
        # 处理当前节点
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
```

### 动态规划模板
```python
def dp_problem(nums):
    n = len(nums)
    # dp[i] 表示状态i的最优解
    dp = [0] * n
    
    # 初始化
    dp[0] = initial_value
    
    # 状态转移
    for i in range(1, n):
        dp[i] = max/min(dp[i-1] + nums[i], dp[i-2] + nums[i], ...)
    
    return dp[n-1]
```

## 📝 机考注意事项

1. **环境准备**
   - 确认编程语言（Python/Java/C++等）
   - 熟悉IDE快捷键
   - 准备常用库的导入语句

2. **代码规范**
   - 变量命名清晰
   - 关键逻辑添加注释
   - 代码结构清晰

3. **调试技巧**
   - 使用print语句调试
   - 画图理解复杂逻辑
   - 先写测试用例再编码

4. **心态调整**
   - 保持冷静，先易后难
   - 遇到难题先跳过，完成其他题目
   - 时间管理很重要

## 🚀 快速练习

建议在机考前：
- [ ] 复习常见算法模板
- [ ] 刷5-10道中等难度题目
- [ ] 练习时间限制下的编码
- [ ] 复习已准备的题目

## 📖 参考资料

- LeetCode 模板：`/root/data/AI/flyllm/leetcode/模板.md`
- 已准备题目：`/root/data/AI/flyllm/leetcode/`

