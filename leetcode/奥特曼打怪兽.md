# 奥特曼打怪兽 🦸

## 📋 题目描述

从 **(0,0)** 到 **(N-1, N-1)**，求能打败最多怪兽的路径。

**网格元素：**
- `0`：空地，可以通过
- `1`：怪兽，打败后通过，获得 **1 分**
- `-1`：墙壁，不能通过，必须绕开

**移动规则：**
- ✅ 只能向**右**或向**下**移动
- ❌ 不能回退、不能斜走

**重要约束：**
- ⚠️ 如果某条路径无法到达终点，不管路上有多少怪兽，都不可选

---

## 🎯 算法分析

### 为什么选择动态规划？

| 特性 | 说明 | 为何适合 DP |
|------|------|------------|
| **最优子结构** | 到达 (i,j) 的最优解由 (i-1,j) 或 (i,j-1) 推导 | ✅ 可以递推 |
| **无后效性** | 只能向右/下，不会回头，无环路 | ✅ 状态独立 |
| **重叠子问题** | 多条路径可能到达同一格子 | ✅ 可以记忆化 |

### 状态定义

```
dp[i][j] = 从起点 (0,0) 到位置 (i,j) 能打败的最多怪兽数
```

### 状态转移方程

```python
if grid[i][j] == -1:  # 墙壁
    dp[i][j] = -∞  # 不可达
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + (1 if grid[i][j] == 1 else 0)
    
# 特殊情况：两个方向都不可达
if dp[i-1][j] == -∞ and dp[i][j-1] == -∞:
    dp[i][j] = -∞
```

### 复杂度分析

- **时间复杂度**：O(N²) - 遍历每个格子一次
- **空间复杂度**：O(N²) - DP 数组，可优化到 O(N)

---

## 💻 Python 实现

### 方法一：标准动态规划（推荐）⭐

```python
def maxMonsters(grid):
    """
    动态规划求最多怪兽数
    
    Args:
        grid: N×N 的二维数组
        
    Returns:
        最多能打败的怪兽数，无法到达返回 -1
    """
    if not grid or not grid[0]:
        return -1
    
    n = len(grid)
    
    # 起点或终点是墙壁，无法完成
    if grid[0][0] == -1 or grid[n-1][n-1] == -1:
        return -1
    
    # 初始化 DP 数组，-∞ 表示不可达
    INF = float('-inf')
    dp = [[INF] * n for _ in range(n)]
    
    # 初始化起点
    dp[0][0] = 1 if grid[0][0] == 1 else 0
    
    # 填充 DP 表
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue
                
            if grid[i][j] == -1:  # 墙壁，不可达
                dp[i][j] = INF
                continue
            
            # 当前格子的怪兽数
            monsters = 1 if grid[i][j] == 1 else 0
            
            # 从上方来
            from_up = dp[i-1][j] if i > 0 else INF
            # 从左方来
            from_left = dp[i][j-1] if j > 0 else INF
            
            # 取最优路径
            if from_up == INF and from_left == INF:
                dp[i][j] = INF  # 两个方向都不可达
            else:
                dp[i][j] = max(from_up, from_left) + monsters
    
    # 返回结果
    return dp[n-1][n-1] if dp[n-1][n-1] != INF else -1
```

**测试代码：**

```python
def test():
    # 测试用例 1：简单路径
    grid1 = [
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]
    ]
    print(f"测试1: {maxMonsters(grid1)}")  # 输出: 2
    
    # 测试用例 2：需要绕路
    grid2 = [
        [1, 1, 1],
        [1, -1, 1],
        [1, 1, 1]
    ]
    print(f"测试2: {maxMonsters(grid2)}")  # 输出: 5
    
    # 测试用例 3：无法到达
    grid3 = [
        [0, -1],
        [-1, 0]
    ]
    print(f"测试3: {maxMonsters(grid3)}")  # 输出: -1
    
    # 测试用例 4：2×2 网格
    grid4 = [
        [1, 1],
        [1, 1]
    ]
    print(f"测试4: {maxMonsters(grid4)}")  # 输出: 3 (只能经过3个格子)

if __name__ == '__main__':
    test()
```

---

### 方法二：空间优化版（O(N) 空间）

```python
def maxMonstersOptimized(grid):
    """
    空间优化的动态规划
    只需要保存上一行的状态
    """
    if not grid or not grid[0]:
        return -1
    
    n = len(grid)
    
    if grid[0][0] == -1 or grid[n-1][n-1] == -1:
        return -1
    
    INF = float('-inf')
    
    # 只需要两行
    prev = [INF] * n
    curr = [INF] * n
    
    prev[0] = 1 if grid[0][0] == 1 else 0
    
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                curr[j] = prev[0]
                continue
            
            if grid[i][j] == -1:
                curr[j] = INF
                continue
            
            monsters = 1 if grid[i][j] == 1 else 0
            
            from_up = prev[j] if i > 0 else INF
            from_left = curr[j-1] if j > 0 else INF
            
            if from_up == INF and from_left == INF:
                curr[j] = INF
            else:
                curr[j] = max(from_up, from_left) + monsters
        
        prev, curr = curr, prev
    
    return prev[n-1] if prev[n-1] != INF else -1
```

---

### 方法三：带路径记录版

```python
def maxMonstersWithPath(grid):
    """
    返回最多怪兽数和具体路径
    """
    if not grid or not grid[0]:
        return -1, []
    
    n = len(grid)
    
    if grid[0][0] == -1 or grid[n-1][n-1] == -1:
        return -1, []
    
    INF = float('-inf')
    dp = [[INF] * n for _ in range(n)]
    path = [[None] * n for _ in range(n)]  # 记录来自哪个方向
    
    dp[0][0] = 1 if grid[0][0] == 1 else 0
    
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue
                
            if grid[i][j] == -1:
                dp[i][j] = INF
                continue
            
            monsters = 1 if grid[i][j] == 1 else 0
            
            from_up = dp[i-1][j] if i > 0 else INF
            from_left = dp[i][j-1] if j > 0 else INF
            
            if from_up == INF and from_left == INF:
                dp[i][j] = INF
            else:
                if from_up >= from_left:
                    dp[i][j] = from_up + monsters
                    path[i][j] = 'DOWN'  # 从上方来
                else:
                    dp[i][j] = from_left + monsters
                    path[i][j] = 'RIGHT'  # 从左方来
    
    if dp[n-1][n-1] == INF:
        return -1, []
    
    # 回溯路径
    result_path = []
    i, j = n-1, n-1
    
    while i != 0 or j != 0:
        result_path.append((i, j))
        if path[i][j] == 'DOWN':
            i -= 1
        else:  # 'RIGHT'
            j -= 1
    
    result_path.append((0, 0))
    result_path.reverse()
    
    return dp[n-1][n-1], result_path


# 测试带路径版本
def testWithPath():
    grid = [
        [1, 1, 0],
        [0, -1, 1],
        [1, 1, 1]
    ]
    
    monsters, path = maxMonstersWithPath(grid)
    print(f"最多怪兽数: {monsters}")
    print(f"路径: {path}")
    
    # 可视化路径
    n = len(grid)
    path_grid = [['.' for _ in range(n)] for _ in range(n)]
    for i, j in path:
        if grid[i][j] == 1:
            path_grid[i][j] = 'M'  # Monster
        elif grid[i][j] == -1:
            path_grid[i][j] = 'X'  # Wall
        else:
            path_grid[i][j] = 'O'  # Path
    
    print("\n路径可视化 (M=怪兽, X=墙壁, O=路径):")
    for row in path_grid:
        print(' '.join(row))

if __name__ == '__main__':
    testWithPath()
```

---

## 🔄 其他算法对比

### DFS + 记忆化（本质也是 DP）

```python
def maxMonstersDFS(grid):
    """
    DFS + 记忆化（效果等同于 DP，但从终点往起点搜索）
    """
    if not grid or not grid[0]:
        return -1
    
    n = len(grid)
    memo = {}
    
    def dfs(i, j):
        # 越界
        if i >= n or j >= n:
            return float('-inf')
        
        # 墙壁
        if grid[i][j] == -1:
            return float('-inf')
        
        # 终点
        if i == n-1 and j == n-1:
            return 1 if grid[i][j] == 1 else 0
        
        # 记忆化
        if (i, j) in memo:
            return memo[(i, j)]
        
        # 当前格子怪兽数
        monsters = 1 if grid[i][j] == 1 else 0
        
        # 向右或向下
        right = dfs(i, j+1)
        down = dfs(i+1, j)
        
        # 都不可达
        if right == float('-inf') and down == float('-inf'):
            result = float('-inf')
        else:
            result = max(right, down) + monsters
        
        memo[(i, j)] = result
        return result
    
    result = dfs(0, 0)
    return result if result != float('-inf') else -1
```

---

## 📊 方法对比总结

| 方法 | 时间复杂度 | 空间复杂度 | 优点 | 缺点 | 推荐度 |
|------|-----------|-----------|------|------|--------|
| **标准 DP** | O(N²) | O(N²) | 清晰易懂、易扩展 | 需要额外空间 | ⭐⭐⭐⭐⭐ |
| **DP 空间优化** | O(N²) | O(N) | 节省空间 | 无法直接记录路径 | ⭐⭐⭐⭐ |
| **带路径记录** | O(N²) | O(N²) | 可以得到具体路径 | 代码稍复杂 | ⭐⭐⭐⭐ |
| **DFS + 记忆化** | O(N²) | O(N²) | 代码直观 | 递归深度限制 | ⭐⭐⭐ |
| **纯 DFS/回溯** | O(2^N²) | O(N) | 实现简单 | 效率极低 | ⭐ |

**推荐使用标准动态规划方法**，因为：
- ✅ 代码清晰易懂
- ✅ 效率最优 O(N²)
- ✅ 易于扩展（如记录路径）
- ✅ 面试常见解法

---

## ⚠️ 重要约束：不可达路径处理

### 问题说明

即使某条路径上有很多怪兽，如果**无法到达终点**，这条路径及其怪兽都**不计入结果**。

### 示例演示

```
网格：
[1, 1, 1, -1]    ← 上方有 3 只怪兽
[0, 0, -1, 1]
[0, 0, 1, 1]     ← 下方有 2 只怪兽
[0, 0, 0, 0]

DP 状态传播过程：
 1  2  3  X      ← 墙壁阻断，后续全部不可达 (X = -∞)
 1  2  X  X
 1  2  3  4      ← 只能走下方路径
 1  2  3  4      ← 终点可达

结果：4 (只计算可达路径)
```

**为什么上方的 3 只怪兽不算？**
- 虽然上方路径有 3 只怪兽（比下方的 2 只多）
- 但是被墙壁堵死，**无法到达终点** (3,3)
- 算法正确选择下方可达路径：0→右→右→下→(1,2)怪兽→(1,3)怪兽→下→下→终点
- 总共：2 只（下方）+ 2 只（右侧）= 4 只怪兽

### 实现机制

```python
# 1. 墙壁位置标记为 -∞
if grid[i][j] == -1:
    dp[i][j] = float('-inf')

# 2. 如果上方和左方都不可达，当前位置也不可达
if from_up == INF and from_left == INF:
    dp[i][j] = INF  # 不可达状态会"传播"

# 3. 最后检查终点是否可达
return dp[n-1][n-1] if dp[n-1][n-1] != INF else -1
```

### 状态传播特性

- **可达性传播**：只有从可达位置出发才能到达新位置
- **自动过滤**：不可达区域（-∞）在 `max()` 比较中被自动排除
- **隔离效应**：死胡同中的怪兽数量不会"污染"主路径的计算

**验证测试：**
运行 `test_unreachable.py` 可查看完整的不可达路径测试用例。

---

## 🎓 关键知识点

### 1. 边界处理
- 起点或终点是墙壁 → 直接返回 -1
- 第一行/第一列需要特殊初始化

### 2. 不可达标记
- 使用 `float('-inf')` 标记不可达状态
- 便于在 `max()` 中自动过滤

### 3. 状态转移逻辑
```python
dp[i][j] = max(
    dp[i-1][j],  # 从上方来
    dp[i][j-1]   # 从左方来
) + monsters
```

### 4. 结果验证
- 最后必须检查 `dp[n-1][n-1] != -∞`
- 确保返回的是可达路径的结果

---

## 🚀 扩展思考

如果题目变为以下情况，如何调整算法？

| 变化 | 算法调整 |
|------|---------|
| **可以向四个方向移动** | 使用 BFS + 状态记录，可能有环需要访问标记 |
| **有多个起点/终点** | 对每个起点运行 DP，取最大值 |
| **怪兽有不同分数** | 修改 `monsters = grid[i][j]` |
| **有怪兽血量限制** | 状态定义改为 `dp[i][j][hp]` (位置+剩余血量) |
| **可以后退但不能重复** | DFS + 回溯 + 访问标记 |
| **求所有可行路径** | DFS 遍历 + 路径记录 |

---

## 📁 相关文件

- **完整测试**：`奥特曼打怪兽_test.py` - 基础功能测试
- **专项测试**：`test_unreachable.py` - 不可达路径测试
- **使用方法**：
  ```bash
  python 奥特曼打怪兽_test.py        # 运行基础测试
  python test_unreachable.py         # 验证不可达处理
  ```

---

## 📝 总结

这道题是**典型的动态规划问题**，核心在于：

1. ✅ **正确识别**：单向路径 + 最优子结构 → DP
2. ✅ **状态定义**：`dp[i][j]` = 到达 (i,j) 的最多怪兽数
3. ✅ **状态转移**：从上方或左方的最优解推导
4. ✅ **边界处理**：墙壁用 -∞ 标记，自动传播不可达状态
5. ✅ **结果验证**：检查终点是否可达

**时间复杂度 O(N²)**，**空间复杂度 O(N²)**（可优化到 O(N)）

这是面试中的经典题型，掌握好动态规划的思路即可轻松解决！🎯
