import heapq
from typing import List, Optional


class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None


class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        """
        合并K个有序链表
        时间复杂度: O(N log K)，其中N是所有链表中的节点总数，K是链表个数
        空间复杂度: O(K)，优先队列中最多有K个节点
        """
        # 创建一个虚拟头节点
        dummy = ListNode(0)
        current = dummy
        
        # 创建优先队列，存储每个链表的头节点
        # 使用(节点值, 链表索引, 节点)作为元组，确保相同值的节点可以正确比较
        pq = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(pq, (node.val, i, node))
        
        # 当优先队列不为空时，不断取出最小节点
        while pq:
            # 取出当前最小节点
            val, i, node = heapq.heappop(pq)
            
            # 将节点添加到结果链表中
            current.next = node
            current = current.next
            
            # 如果该链表还有下一个节点，将其加入优先队列
            if node.next:
                heapq.heappush(pq, (node.next.val, i, node.next))
        
        return dummy.next


def create_linked_list(values: List[int]) -> ListNode:
    """创建链表"""
    dummy = ListNode(0)
    current = dummy
    for val in values:
        current.next = ListNode(val)
        current = current.next
    return dummy.next


def print_linked_list(head: ListNode) -> None:
    """打印链表"""
    values = []
    while head:
        values.append(str(head.val))
        head = head.next
    print(" -> ".join(values))


def test_merge_k_lists():
    """测试函数"""
    # 创建测试用例
    list1 = create_linked_list([1, 4, 5])
    list2 = create_linked_list([1, 3, 4])
    list3 = create_linked_list([2, 6])
    
    # 打印原始链表
    print("原始链表:")
    print("List 1:", end=" ")
    print_linked_list(list1)
    print("List 2:", end=" ")
    print_linked_list(list2)
    print("List 3:", end=" ")
    print_linked_list(list3)
    
    # 合并链表
    solution = Solution()
    merged_list = solution.mergeKLists([list1, list2, list3])
    
    # 打印合并后的链表
    print("\n合并后的链表:")
    print_linked_list(merged_list)


if __name__ == "__main__":
    test_merge_k_lists()
