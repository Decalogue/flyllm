```python
import sys

ss = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@_'
s2n = {k: i+10 for i,k in enumerate(ss)}


def bash2num(s):
    if len(s) == 1:
        try:
            return int(s)
        except:
            return 'ERROR'

    if '#' not in s:
        if s.startswith('0') and len(s) > 1:
            # 八进制
            if all([k in '01234567' for k in s[1:]]):  # 检查0后面的字符
                res = 0
                for k in s[1:]:
                    res *= 8
                    res += int(k)
                return res
            # 十六进制
            # 检查所有字符是否都在十六进制范围内（0-9, a-f, A-F）
            if all([k in '0123456789abcdefABCDEF' for k in s[1:]]):
                res = 0
                for k in s[1:]:
                    res *= 16
                    # 十六进制转换：0-9直接转，a-f对应10-15，A-F对应10-15
                    if k.isdigit():
                        res += int(k)
                    elif k in 'abcdef':
                        res += ord(k) - ord('a') + 10
                    else:  # 'ABCDEF'
                        res += ord(k) - ord('A') + 10
                return res
            return 'ERROR'
        try:
            return int(s)
        except:
            return 'ERROR'
    else:
        base, s = s.split('#')
        if '#' in s:
            return 'ERROR'
        try:
            res = 0
            for k in s:
                res *= int(base)
                res += s2n[k] if k in ss else int(k)
            return res
        except:
            return 'ERROR'


for line in sys.stdin:
    a = line.split()[0]
    num = bash2num(a)
    print(f'{a} = {num}')
```