class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, n in enumerate(nums):
            if target-n in d:
                return [i, d[target-n]]
            d[n] = i
        return []