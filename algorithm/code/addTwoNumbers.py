class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        c = 0
        res = None
        p = None
        while l1 and l2:
            n = l1.val + l2.val + c
            if n > 9:
                n = n - 10
                c = 1
            else:
                c = 0
            if not res:
                res = ListNode(n)
                p = res
            else:
                p.next = ListNode(n)
                p = p.next
            l1 = l1.next
            l2 = l2.next
        l = l1 if l1 else l2
        if c > 0:
            if l:
                while l:
                    n = l.val + c
                    if n > 9:
                        n = n - 10
                        c = 1
                    else:
                        c = 0
                    p.next = ListNode(n)
                    p = p.next
                    l = l.next
                if c > 0:
                    p.next = ListNode(c)
            else:
                p.next = ListNode(c)
        else:
            if l:
                p.next = l
        return res