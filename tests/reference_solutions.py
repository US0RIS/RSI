"""Reference solutions used only to validate the built-in benchmark checks."""

REFERENCE = {}

REFERENCE["decode_nested"] = '''
def decode_nested(s):
    stack = [["", 1]]
    num = 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == "[":
            stack.append(["", num]); num = 0
        elif ch == "]":
            text, k = stack.pop()
            stack[-1][0] += text * k
        else:
            stack[-1][0] += ch
    return stack[0][0]
'''

REFERENCE["eval_expr"] = '''
def eval_expr(s):
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            toks.append(int(s[i:j])); i = j
        else:
            toks.append(c); i += 1
    pos = 0
    def peek():
        return toks[pos] if pos < len(toks) else None
    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]
    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take(); r = term()
            v = v + r if op == "+" else v - r
        return v
    def term():
        v = unary()
        while peek() in ("*", "/"):
            op = take(); r = unary()
            if op == "*":
                v = v * r
            else:
                q = abs(v) // abs(r)
                v = q if (v >= 0) == (r >= 0) else -q
        return v
    def unary():
        if peek() == "-":
            take(); return -unary()
        if peek() == "(":
            take(); v = expr(); take(); return v
        return take()
    return expr()
'''

REFERENCE["merge_intervals"] = '''
def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
'''

REFERENCE["lru_trace"] = '''
from collections import OrderedDict
def lru_trace(capacity, ops):
    d = OrderedDict(); res = []
    for op in ops:
        if op[0] == "get":
            if op[1] in d:
                d.move_to_end(op[1]); res.append(d[op[1]])
            else:
                res.append(-1)
        else:
            if capacity == 0:
                continue
            k, v = op[1], op[2]
            if k in d:
                d.move_to_end(k)
            elif len(d) >= capacity:
                d.popitem(last=False)
            d[k] = v
    return res
'''

REFERENCE["topo_order"] = '''
import heapq
def topo_order(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
    indeg = [0] * n
    for u in range(n):
        for v in adj[u]:
            indeg[v] += 1
    h = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(h); out = []
    while h:
        u = heapq.heappop(h); out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(h, v)
    return out if len(out) == n else None
'''

REFERENCE["to_roman"] = '''
def to_roman(n):
    if type(n) is not int or not 1 <= n <= 3999:
        raise ValueError(n)
    vals = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s; n -= v
    return out
'''

REFERENCE["regex_match"] = '''
from functools import lru_cache
def regex_match(s, p):
    @lru_cache(None)
    def m(i, j):
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], ".")
        if j + 1 < len(p) and p[j + 1] == "*":
            return m(i, j + 2) or (first and m(i + 1, j))
        return first and m(i + 1, j + 1)
    return bool(m(0, 0))
'''

REFERENCE["count_lis"] = '''
def count_lis(nums):
    if not nums:
        return (0, 0)
    n = len(nums); L = [1] * n; C = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                if L[j] + 1 > L[i]:
                    L[i] = L[j] + 1; C[i] = C[j]
                elif L[j] + 1 == L[i]:
                    C[i] += C[j]
    m = max(L)
    return (m, sum(c for l, c in zip(L, C) if l == m))
'''

REFERENCE["osa_distance"] = '''
def osa_distance(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): d[i][0] = i
    for j in range(len(b) + 1): d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + c)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + 1)
    return d[len(a)][len(b)]
'''

REFERENCE["word_wrap"] = '''
def word_wrap(text, width):
    words = []
    for w in text.split():
        while len(w) > width:
            words.append(w[:width]); w = w[width:]
        if w:
            words.append(w)
    lines = []; cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines
'''

REFERENCE["spiral"] = '''
def spiral(matrix):
    out = []
    m = [row[:] for row in matrix if row]
    while m:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1] if m else []
        m = [r for r in m if r]
    return out
'''

REFERENCE["big_multiply"] = '''
def big_multiply(a, b):
    neg = (a.startswith("-")) != (b.startswith("-"))
    a = a.lstrip("-").lstrip("0") or "0"; b = b.lstrip("-").lstrip("0") or "0"
    res = [0] * (len(a) + len(b))
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            p = (ord(a[i]) - 48) * (ord(b[j]) - 48) + res[i + j + 1]
            res[i + j + 1] = p % 10; res[i + j] += p // 10
    s = "".join(map(str, res)).lstrip("0") or "0"
    return "-" + s if neg and s != "0" else s
'''

REFERENCE["count_shortest_paths"] = '''
import heapq
def count_shortest_paths(n, edges, src, dst):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w)); adj[v].append((u, w))
    INF = float("inf"); dist = [INF] * n; cnt = [0] * n
    dist[src] = 0; cnt[src] = 1; h = [(0, src)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; cnt[v] = cnt[u]; heapq.heappush(h, (nd, v))
            elif nd == dist[v]:
                cnt[v] += cnt[u]
    return (-1, 0) if dist[dst] == INF else (dist[dst], cnt[dst])
'''

REFERENCE["parse_csv_line"] = '''
def parse_csv_line(line):
    out = []; cur = ""; i = 0; q = False
    while i < len(line):
        c = line[i]
        if q:
            if c == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur += '"'; i += 1
                else:
                    q = False
            else:
                cur += c
        elif c == '"':
            q = True
        elif c == ",":
            out.append(cur); cur = ""
        else:
            cur += c
        i += 1
    out.append(cur)
    return out
'''

REFERENCE["days_between"] = '''
from datetime import date
def days_between(d1, d2):
    a = date.fromisoformat(d1); b = date.fromisoformat(d2)
    return (b - a).days
'''

REFERENCE["valid_sudoku"] = '''
def valid_sudoku(board):
    if len(board) != 9 or any(len(r) != 9 for r in board):
        return False
    seen = set()
    for i, r in enumerate(board):
        for j, c in enumerate(r):
            if c == ".":
                continue
            if c not in "123456789":
                return False
            keys = (("r", i, c), ("c", j, c), ("b", i // 3, j // 3, c))
            if any(k in seen for k in keys):
                return False
            seen.update(keys)
    return True
'''

REFERENCE["format_duration"] = '''
def format_duration(seconds):
    if seconds == 0:
        return "now"
    parts = []
    for name, size in (("year", 31536000), ("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
        q, seconds = divmod(seconds, size)
        if q:
            parts.append(f"{q} {name}" + ("s" if q > 1 else ""))
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
'''

REFERENCE["min_window"] = '''
from collections import Counter
def min_window(s, t):
    if not t:
        return ""
    need = Counter(t); missing = len(t); best = (float("inf"), 0, 0); l = 0
    for r, c in enumerate(s, 1):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        if missing == 0:
            while need[s[l]] < 0:
                need[s[l]] += 1; l += 1
            if r - l < best[0]:
                best = (r - l, l, r)
            need[s[l]] += 1; missing += 1; l += 1
    return s[best[1]:best[2]] if best[0] != float("inf") else ""
'''

REFERENCE["run_program"] = '''
import re
def run_program(lines):
    env = {}
    for line in lines:
        name, expr = line.split("=", 1)
        toks = re.findall(r"\\d+|[A-Za-z_]\\w*|//|[-+*()]", expr)
        pos = 0
        def peek():
            return toks[pos] if pos < len(toks) else None
        def take():
            nonlocal pos
            pos += 1
            return toks[pos - 1]
        def e():
            v = t()
            while peek() in ("+", "-"):
                op = take(); r = t(); v = v + r if op == "+" else v - r
            return v
        def t():
            v = u()
            while peek() in ("*", "//"):
                op = take(); r = u()
                v = v * r if op == "*" else v // r
            return v
        def u():
            if peek() == "-":
                take(); return -u()
            if peek() == "(":
                take(); v = e(); take(); return v
            tok = take()
            if tok.isdigit():
                return int(tok)
            if tok not in env:
                raise NameError(tok)
            return env[tok]
        env[name.strip()] = e()
    return env
'''

REFERENCE["knapsack"] = '''
def knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]
'''

REFERENCE["word_break_all"] = '''
from functools import lru_cache
def word_break_all(s, words):
    ws = set(words)
    @lru_cache(None)
    def go(i):
        if i == len(s):
            return [[]]
        out = []
        for j in range(i + 1, len(s) + 1):
            if s[i:j] in ws:
                out += [[s[i:j]] + rest for rest in go(j)]
        return out
    return sorted(" ".join(p) for p in go(0))
'''

REFERENCE["next_permutation"] = '''
def next_permutation(nums):
    a = list(nums); i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a
'''

REFERENCE["longest_valid_parens"] = '''
def longest_valid_parens(s):
    st = [-1]; best = 0
    for i, c in enumerate(s):
        if c == "(":
            st.append(i)
        else:
            st.pop()
            if not st:
                st.append(i)
            else:
                best = max(best, i - st[-1])
    return best
'''

REFERENCE["islands"] = '''
def islands(grid):
    seen = set(); count = 0; best = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "1" and (i, j) not in seen:
                count += 1; size = 0; st = [(i, j)]; seen.add((i, j))
                while st:
                    x, y = st.pop(); size += 1
                    for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                        if 0 <= nx < len(grid) and 0 <= ny < len(grid[nx]) and grid[nx][ny] == "1" and (nx, ny) not in seen:
                            seen.add((nx, ny)); st.append((nx, ny))
                best = max(best, size)
    return (count, best)
'''

REFERENCE["kth_pair_distance"] = '''
def kth_pair_distance(nums, k):
    a = sorted(nums)
    lo, hi = 0, a[-1] - a[0]
    while lo < hi:
        mid = (lo + hi) // 2
        cnt = 0; l = 0
        for r in range(len(a)):
            while a[r] - a[l] > mid:
                l += 1
            cnt += r - l
        if cnt >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
'''

REFERENCE["justify"] = '''
def justify(words, width):
    lines = []; cur = []
    for w in words:
        if cur and len(" ".join(cur + [w])) > width:
            lines.append(cur); cur = []
        cur.append(w)
    out = []
    for line in lines:
        if len(line) == 1:
            out.append(line[0].ljust(width)); continue
        gaps = len(line) - 1; spaces = width - sum(map(len, line))
        q, r = divmod(spaces, gaps); s = ""
        for i, w in enumerate(line[:-1]):
            s += w + " " * (q + (1 if i < r else 0))
        out.append(s + line[-1])
    out.append(" ".join(cur).ljust(width))
    return out
'''

REFERENCE["parse_duration"] = '''
import re
def parse_duration(s):
    m = re.fullmatch(r"(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", s)
    if not s or not m:
        raise ValueError(s)
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
'''

REFERENCE["top_k_suggestions"] = '''
def top_k_suggestions(freq, prefix, k):
    c = [w for w in freq if w.startswith(prefix)]
    return sorted(c, key=lambda w: (-freq[w], w))[:k]
'''

REFERENCE["rotate_cw"] = '''
def rotate_cw(matrix):
    if not matrix:
        return []
    return [list(r) for r in zip(*matrix[::-1])]
'''

REFERENCE["sum_fractions"] = '''
from fractions import Fraction
def sum_fractions(fracs):
    t = Fraction(0)
    for f in fracs:
        a, b = f.split("/")
        t += Fraction(int(a), int(b))
    return f"{t.numerator}/{t.denominator}"
'''
