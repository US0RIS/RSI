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

# ------------------------------------------------------------------ audit split
REFERENCE["roman_to_int"] = '''
def roman_to_int(s):
    vals = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
    def to_r(n):
        out = ""
        for v, r in vals:
            while n >= v:
                out += r; n -= v
        return out
    if not isinstance(s, str) or not s:
        raise ValueError(s)
    single = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total = 0
    for i, ch in enumerate(s):
        if ch not in single:
            raise ValueError(s)
        v = single[ch]
        total += -v if i + 1 < len(s) and single.get(s[i + 1], 0) > v else v
    if not 1 <= total <= 3999 or to_r(total) != s:
        raise ValueError(s)
    return total
'''

REFERENCE["group_anagrams"] = '''
def group_anagrams(words):
    g = {}
    for w in words:
        g.setdefault("".join(sorted(w)), []).append(w)
    return sorted((sorted(v) for v in g.values()), key=lambda grp: grp[0])
'''

REFERENCE["max_subarray"] = '''
def max_subarray(nums):
    best = None
    for i in range(len(nums)):
        t = 0
        for j in range(i, len(nums)):
            t += nums[j]
            if best is None or t > best[0]:
                best = (t, i, j)
    return best
'''

REFERENCE["flatten_json"] = '''
def flatten_json(obj):
    out = {}
    def go(x, path):
        if isinstance(x, dict) and x:
            for k, v in x.items():
                go(v, path + [str(k)])
        elif isinstance(x, list) and x:
            for i, v in enumerate(x):
                go(v, path + [str(i)])
        else:
            out[".".join(path)] = x
    for k, v in obj.items():
        go(v, [str(k)])
    return out
'''

REFERENCE["compress_ranges"] = '''
def compress_ranges(nums):
    a = sorted(set(nums)); out = []; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[j + 1] == a[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{a[i]}-{a[j]}")
        else:
            out += [str(x) for x in a[i:j + 1]]
        i = j + 1
    return ",".join(out)
'''

REFERENCE["word_ladder"] = '''
from collections import deque
def word_ladder(begin, end, words):
    if begin == end:
        return 1
    ws = set(words); q = deque([(begin, 1)]); seen = {begin}
    while q:
        w, d = q.popleft()
        for i in range(len(w)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                n = w[:i] + c + w[i + 1:]
                if n in ws and n not in seen:
                    if n == end:
                        return d + 1
                    seen.add(n); q.append((n, d + 1))
    return 0
'''

REFERENCE["grid_paths"] = '''
def grid_paths(grid):
    if not grid or not grid[0] or grid[0][0] == "#" or grid[-1][-1] == "#":
        return 0
    w = len(grid[0]); row = [0] * w
    for r, line in enumerate(grid):
        for c in range(w):
            if line[c] == "#":
                row[c] = 0
            elif r == 0 and c == 0:
                row[c] = 1
            else:
                row[c] = row[c] + (row[c - 1] if c else 0)
    return row[-1]
'''

REFERENCE["eval_rpn"] = '''
def eval_rpn(tokens):
    st = []
    for t in tokens:
        if t in ("+", "-", "*", "/"):
            if len(st) < 2:
                raise ValueError(t)
            b = st.pop(); a = st.pop()
            if t == "+": st.append(a + b)
            elif t == "-": st.append(a - b)
            elif t == "*": st.append(a * b)
            else:
                q = abs(a) // abs(b)
                st.append(q if (a >= 0) == (b >= 0) else -q)
        else:
            try:
                st.append(int(t))
            except ValueError:
                raise ValueError(t) from None
    if len(st) != 1:
        raise ValueError(tokens)
    return st[0]
'''

REFERENCE["min_rooms"] = '''
def min_rooms(meetings):
    ev = []
    for s, e in meetings:
        if e > s:
            ev += [(s, 1), (e, -1)]
    best = cur = 0
    for _, d in sorted(ev, key=lambda x: (x[0], x[1])):
        cur += d; best = max(best, cur)
    return best
'''

REFERENCE["base_convert"] = '''
def base_convert(s, from_base, to_base):
    if not (2 <= from_base <= 36 and 2 <= to_base <= 36):
        raise ValueError("base")
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    neg = s.startswith("-"); body = s[1:] if neg else s
    if not body:
        raise ValueError(s)
    n = 0
    for ch in body.upper():
        d = digits.find(ch)
        if d < 0 or d >= from_base:
            raise ValueError(s)
        n = n * from_base + d
    if n == 0:
        return "0"
    out = ""
    while n:
        n, r = divmod(n, to_base); out = digits[r] + out
    return "-" + out if neg else out
'''

REFERENCE["scs_length"] = '''
def scs_length(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return len(a) + len(b) - prev[-1]
'''

REFERENCE["cron_match"] = '''
def cron_match(expr, minute, hour):
    def field(spec, value, lo, hi):
        for item in spec.split(","):
            rng, _, step = item.partition("/")
            step = int(step) if step else 1
            if rng == "*":
                a, b = lo, hi
            elif "-" in rng:
                a, b = map(int, rng.split("-"))
            else:
                a = b = int(rng)
            if a <= value <= b and (value - a) % step == 0:
                return True
        return False
    m, h = expr.split()
    return field(m, minute, 0, 59) and field(h, hour, 0, 23)
'''
