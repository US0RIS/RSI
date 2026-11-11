"""Built-in benchmark: 30 original Python function-implementation tasks.

Each task has a visible ``prompt`` (what the agent sees) and hidden ``checks``
(boolean Python expressions, evaluated only by the supervisor's grader). The
agent never receives the checks. ``split`` is "dev" (failures are summarised
back to the self-improvement step) or "holdout" (only aggregate scores are
used, to decide promotion; never shown to any genome) or "audit" (measurement only).

``_raises(Exc, fn, *args)`` is available inside checks.
"""

TASKS = [
    # ------------------------------------------------------------------ dev
    {
        "id": "decode_nested", "split": "dev", "entry_point": "decode_nested",
        "prompt": "Write `decode_nested(s: str) -> str`. The input is an encoded string where `k[...]` means the "
                  "bracketed part repeated k times (k is a positive integer, possibly multi-digit). Brackets can be "
                  "nested and letters may appear outside brackets. Return the decoded string. "
                  "Example: decode_nested('3[a2[c]]') == 'accaccacc'.",
        "checks": [
            "decode_nested('3[a2[c]]') == 'accaccacc'",
            "decode_nested('2[abc]3[cd]ef') == 'abcabccdcdcdef'",
            "decode_nested('') == ''",
            "decode_nested('abc') == 'abc'",
            "decode_nested('10[a]') == 'a' * 10",
            "decode_nested('x2[y3[z]]w') == 'xyzzzyzzzw'",
            "decode_nested('1[1[1[q]]]') == 'q'",
        ],
    },
    {
        "id": "eval_expr", "split": "dev", "entry_point": "eval_expr",
        "prompt": "Write `eval_expr(s: str) -> int` that evaluates an integer arithmetic expression containing "
                  "non-negative integer literals, `+ - * /`, parentheses, unary minus and arbitrary spaces. `*` and "
                  "`/` bind tighter than `+` and `-`; operators of equal precedence are left-associative. `/` is "
                  "integer division that truncates toward zero (like C). Do not use eval().",
        "checks": [
            "eval_expr('1 + 2 * 3') == 7",
            "eval_expr('(1 + 2) * 3') == 9",
            "eval_expr('7 / 2') == 3",
            "eval_expr('-7 / 2') == -3",
            "eval_expr('7 / -2') == -3",
            "eval_expr('10 - 4 - 3') == 3",
            "eval_expr('2 * (3 + 4) - -5') == 19",
            "eval_expr('  42 ') == 42",
            "eval_expr('-(2 + 3) * 2') == -10",
            "eval_expr('100 / 10 / 3') == 3",
        ],
    },
    {
        "id": "merge_intervals", "split": "dev", "entry_point": "merge_intervals",
        "prompt": "Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]`. Each interval is "
                  "[start, end] with start <= end, in any order. Merge all overlapping intervals; intervals that "
                  "only touch (e.g. [1,2] and [2,3]) also merge. Return the merged intervals sorted by start. "
                  "Do not mutate the input.",
        "checks": [
            "merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
            "merge_intervals([[1,2],[2,3]]) == [[1,3]]",
            "merge_intervals([]) == []",
            "merge_intervals([[5,5]]) == [[5,5]]",
            "merge_intervals([[8,10],[1,4],[3,5]]) == [[1,5],[8,10]]",
            "merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]",
            "(lambda x: (merge_intervals(x), x)[1] == [[3,4],[1,2]])([[3,4],[1,2]])",
        ],
    },
    {
        "id": "lru_trace", "split": "dev", "entry_point": "lru_trace",
        "prompt": "Write `lru_trace(capacity: int, ops: list[tuple]) -> list[int]` simulating an LRU cache. Each op "
                  "is ('get', key) or ('put', key, value). 'get' returns the value or -1 if absent, and marks the "
                  "key most recently used. 'put' inserts/updates and marks it most recently used; if inserting a "
                  "new key exceeds capacity, evict the least recently used key first. With capacity 0 nothing is "
                  "ever stored. Return the list of results of all 'get' ops in order.",
        "checks": [
            "lru_trace(2, [('put',1,1),('put',2,2),('get',1),('put',3,3),('get',2),('put',4,4),('get',1),('get',3),('get',4)]) == [1,-1,-1,3,4]",
            "lru_trace(0, [('put',1,1),('get',1)]) == [-1]",
            "lru_trace(1, [('put',1,1),('put',1,5),('get',1)]) == [5]",
            "lru_trace(2, [('put',1,1),('put',2,2),('put',1,10),('put',3,3),('get',2),('get',1)]) == [-1,10]",
            "lru_trace(3, []) == []",
        ],
    },
    {
        "id": "topo_order", "split": "dev", "entry_point": "topo_order",
        "prompt": "Write `topo_order(n: int, edges: list[tuple[int,int]]) -> list[int] | None`. Nodes are 0..n-1; an "
                  "edge (u, v) means u must come before v. Return the lexicographically smallest topological order, "
                  "or None if the graph has a cycle. Duplicate edges may appear.",
        "checks": [
            "topo_order(4, [(1,0),(2,0),(3,1)]) == [2,3,1,0]",
            "topo_order(3, []) == [0,1,2]",
            "topo_order(2, [(0,1),(1,0)]) is None",
            "topo_order(1, [(0,0)]) is None",
            "topo_order(3, [(0,1),(0,1),(1,2)]) == [0,1,2]",
            "topo_order(0, []) == []",
            "topo_order(5, [(4,3),(3,2),(2,1),(1,0)]) == [4,3,2,1,0]",
        ],
    },
    {
        "id": "to_roman", "split": "dev", "entry_point": "to_roman",
        "prompt": "Write `to_roman(n: int) -> str` that converts an integer in 1..3999 to standard Roman numerals "
                  "(subtractive notation, e.g. 4 -> 'IV', 1994 -> 'MCMXCIV'). Raise ValueError for any other value, "
                  "including non-int types and bool.",
        "checks": [
            "to_roman(1) == 'I'", "to_roman(4) == 'IV'", "to_roman(1994) == 'MCMXCIV'",
            "to_roman(3999) == 'MMMCMXCIX'", "to_roman(444) == 'CDXLIV'",
            "_raises(ValueError, to_roman, 0)", "_raises(ValueError, to_roman, 4000)",
            "_raises(ValueError, to_roman, 2.0)", "_raises(ValueError, to_roman, True)",
        ],
    },
    {
        "id": "regex_match", "split": "dev", "entry_point": "regex_match",
        "prompt": "Write `regex_match(s: str, p: str) -> bool` implementing full-string matching for a tiny regex "
                  "language: '.' matches any single character, 'x*' matches zero or more of the preceding element "
                  "(which may be '.'), all other characters match themselves. Do not use the re module.",
        "checks": [
            "regex_match('aa', 'a') is False", "regex_match('aa', 'a*') is True",
            "regex_match('ab', '.*') is True", "regex_match('aab', 'c*a*b') is True",
            "regex_match('mississippi', 'mis*is*p*.') is False", "regex_match('', 'a*b*') is True",
            "regex_match('', '') is True", "regex_match('a', '') is False",
            "regex_match('abcd', 'a.c.') is True",
        ],
    },
    {
        "id": "count_lis", "split": "dev", "entry_point": "count_lis",
        "prompt": "Write `count_lis(nums: list[int]) -> tuple[int, int]` returning (length of the longest strictly "
                  "increasing subsequence, number of distinct index-sequences achieving that length). For an empty "
                  "list return (0, 0).",
        "checks": [
            "count_lis([1,3,5,4,7]) == (4, 2)", "count_lis([2,2,2,2,2]) == (1, 5)",
            "count_lis([]) == (0, 0)", "count_lis([1,2,3]) == (3, 1)",
            "count_lis([3,2,1]) == (1, 3)", "count_lis([1,2,4,3,5,4,7,2]) == (5, 3)",
        ],
    },
    {
        "id": "osa_distance", "split": "dev", "entry_point": "osa_distance",
        "prompt": "Write `osa_distance(a: str, b: str) -> int`: the optimal string alignment distance, i.e. the "
                  "minimum number of single-character insertions, deletions, substitutions and transpositions of two "
                  "adjacent characters needed to turn a into b, where no substring is edited more than once.",
        "checks": [
            "osa_distance('ca', 'abc') == 3", "osa_distance('abc', 'acb') == 1",
            "osa_distance('', 'abc') == 3", "osa_distance('kitten', 'sitting') == 3",
            "osa_distance('same', 'same') == 0", "osa_distance('ab', 'ba') == 1",
        ],
    },
    {
        "id": "word_wrap", "split": "dev", "entry_point": "word_wrap",
        "prompt": "Write `word_wrap(text: str, width: int) -> list[str]`. Split text on any whitespace into words "
                  "and greedily pack them into lines separated by single spaces, each at most `width` characters. "
                  "A word longer than width is split into chunks of exactly width characters (the last chunk may be "
                  "shorter) and each chunk is treated as its own word. Return [] for text with no words.",
        "checks": [
            "word_wrap('the quick brown fox', 10) == ['the quick', 'brown fox']",
            "word_wrap('   ', 5) == []",
            "word_wrap('abcdefghij', 4) == ['abcd', 'efgh', 'ij']",
            "word_wrap('a  b\\n c', 3) == ['a b', 'c']",
            "word_wrap('hi abcdefgh yo', 4) == ['hi', 'abcd', 'efgh', 'yo']",
            "word_wrap('one', 3) == ['one']",
        ],
    },
    {
        "id": "spiral", "split": "dev", "entry_point": "spiral",
        "prompt": "Write `spiral(matrix: list[list[int]]) -> list[int]` returning the elements in clockwise spiral "
                  "order starting at the top-left. The matrix may be rectangular or empty.",
        "checks": [
            "spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
            "spiral([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]",
            "spiral([]) == []", "spiral([[]]) == []", "spiral([[1],[2],[3]]) == [1,2,3]",
            "spiral([[1,2,3]]) == [1,2,3]",
        ],
    },
    {
        "id": "big_multiply", "split": "dev", "entry_point": "big_multiply",
        "prompt": "Write `big_multiply(a: str, b: str) -> str` that multiplies two decimal integers given as strings "
                  "(optional leading '-', possibly with leading zeros) and returns the product as a canonical string "
                  "(no leading zeros, '0' never negative). Implement the multiplication digit by digit; do not "
                  "convert the whole strings with int().",
        "checks": [
            "big_multiply('12', '34') == '408'", "big_multiply('-12', '34') == '-408'",
            "big_multiply('-12', '-34') == '408'", "big_multiply('0', '-5') == '0'",
            "big_multiply('000123', '0010') == '1230'",
            "big_multiply('123456789123456789', '987654321987654321') == str(123456789123456789*987654321987654321)",
            "big_multiply('-0', '0') == '0'",
        ],
    },
    {
        "id": "count_shortest_paths", "split": "dev", "entry_point": "count_shortest_paths",
        "prompt": "Write `count_shortest_paths(n: int, edges: list[tuple[int,int,int]], src: int, dst: int) -> "
                  "tuple[int, int]`. The graph is undirected with nodes 0..n-1 and positive weights (u, v, w); "
                  "parallel edges may exist and each counts as a distinct path. Return (shortest distance, number "
                  "of distinct shortest paths). If dst is unreachable return (-1, 0). If src == dst return (0, 1).",
        "checks": [
            "count_shortest_paths(4, [(0,1,1),(0,2,1),(1,3,1),(2,3,1)], 0, 3) == (2, 2)",
            "count_shortest_paths(3, [(0,1,5)], 0, 2) == (-1, 0)",
            "count_shortest_paths(2, [(0,1,3),(0,1,3)], 0, 1) == (3, 2)",
            "count_shortest_paths(1, [], 0, 0) == (0, 1)",
            "count_shortest_paths(4, [(0,1,1),(1,2,1),(0,2,2),(2,3,1)], 0, 3) == (3, 2)",
        ],
    },
    {
        "id": "parse_csv_line", "split": "dev", "entry_point": "parse_csv_line",
        "prompt": "Write `parse_csv_line(line: str) -> list[str]` that splits one CSV line on commas. A field may be "
                  "wrapped in double quotes, in which case it may contain commas, and a doubled quote \"\" inside "
                  "it stands for one quote character. Empty fields are kept. Do not use the csv module.",
        "checks": [
            "parse_csv_line('a,b,c') == ['a','b','c']",
            "parse_csv_line('a,,c') == ['a','','c']",
            "parse_csv_line('\"x,y\",z') == ['x,y','z']",
            "parse_csv_line('\"he said \"\"hi\"\"\",2') == ['he said \"hi\"','2']",
            "parse_csv_line('') == ['']", "parse_csv_line(',') == ['','']",
            "parse_csv_line('\"\"') == ['']",
        ],
    },
    {
        "id": "days_between", "split": "dev", "entry_point": "days_between",
        "prompt": "Write `days_between(d1: str, d2: str) -> int` returning the signed number of days from date d1 to "
                  "date d2 (positive if d2 is later). Dates are 'YYYY-MM-DD' in the proleptic Gregorian calendar, "
                  "years 1..9999.",
        "checks": [
            "days_between('2024-01-01', '2024-03-01') == 60", "days_between('2023-01-01', '2023-03-01') == 59",
            "days_between('2000-02-28', '1900-02-28') == -36524", "days_between('2020-05-05', '2020-05-05') == 0",
            "days_between('0001-01-01', '9999-12-31') == 3652058",
        ],
    },
    {
        "id": "valid_sudoku", "split": "dev", "entry_point": "valid_sudoku",
        "prompt": "Write `valid_sudoku(board: list[str]) -> bool`. board is 9 strings of 9 characters each, using "
                  "digits '1'-'9' and '.' for empty cells. Return True iff no row, column or 3x3 box contains a "
                  "repeated digit (the board need not be solvable). Return False if the shape is wrong or any other "
                  "character appears.",
        "checks": [
            "valid_sudoku(['53..7....','6..195...','.98....6.','8...6...3','4..8.3..1','7...2...6','.6....28.','...419..5','....8..79']) is True",
            "valid_sudoku(['83..7....','6..195...','.98....6.','8...6...3','4..8.3..1','7...2...6','.6....28.','...419..5','....8..79']) is False",
            "valid_sudoku(['.........']*9) is True",
            "valid_sudoku(['.........']*8) is False",
            "valid_sudoku(['0........'] + ['.........']*8) is False",
            "valid_sudoku(['1........', '.1.......'] + ['.........']*7) is False",
        ],
    },
    {
        "id": "format_duration", "split": "dev", "entry_point": "format_duration",
        "prompt": "Write `format_duration(seconds: int) -> str` for a non-negative integer. Use units year (365 days), "
                  "day, hour, minute, second; omit zero units; pluralise with 's'; join with ', ' except the last two "
                  "which are joined with ' and '. Return 'now' for 0. Example: 3662 -> '1 hour, 1 minute and 2 seconds'.",
        "checks": [
            "format_duration(0) == 'now'", "format_duration(1) == '1 second'",
            "format_duration(62) == '1 minute and 2 seconds'", "format_duration(3600) == '1 hour'",
            "format_duration(3662) == '1 hour, 1 minute and 2 seconds'",
            "format_duration(132030240) == '4 years, 68 days, 3 hours and 4 minutes'",
        ],
    },
    {
        "id": "min_window", "split": "dev", "entry_point": "min_window",
        "prompt": "Write `min_window(s: str, t: str) -> str` returning the shortest substring of s that contains every "
                  "character of t with at least the same multiplicity. If several shortest windows exist return the "
                  "leftmost. Return '' if none exists or t is empty.",
        "checks": [
            "min_window('ADOBECODEBANC', 'ABC') == 'BANC'", "min_window('a', 'aa') == ''",
            "min_window('aa', 'aa') == 'aa'", "min_window('abc', '') == ''",
            "min_window('abcabc', 'cb') == 'bc'", "min_window('xyz', 'q') == ''",
        ],
    },
    # -------------------------------------------------------------- holdout
    {
        "id": "run_program", "split": "holdout", "entry_point": "run_program",
        "prompt": "Write `run_program(lines: list[str]) -> dict[str, int]`. Each line is `name = expr` where expr "
                  "uses integer literals, previously assigned variable names, + - * and // (floor division, as in "
                  "Python), parentheses and unary minus, with standard precedence. Execute the lines in order and "
                  "return the final variables. Raise NameError for an unassigned variable and ZeroDivisionError for "
                  "division by zero. Do not use eval or exec.",
        "checks": [
            "run_program(['x = 3', 'y = x * (2 + 1)']) == {'x': 3, 'y': 9}",
            "run_program(['a = 7 // 2', 'b = -7 // 2']) == {'a': 3, 'b': -4}",
            "run_program(['a = 1', 'a = a + 1', 'b = -a - -a']) == {'a': 2, 'b': 0}",
            "_raises(NameError, run_program, ['x = y + 1'])",
            "_raises(ZeroDivisionError, run_program, ['x = 1 // (2 - 2)'])",
            "run_program([]) == {}",
            "run_program(['long_name1 = 10', 'z = long_name1 * long_name1 - 1']) == {'long_name1': 10, 'z': 99}",
        ],
    },
    {
        "id": "knapsack", "split": "holdout", "entry_point": "knapsack",
        "prompt": "Write `knapsack(weights: list[int], values: list[int], capacity: int) -> int` returning the maximum "
                  "total value of a subset of items (each used at most once) whose total weight <= capacity.",
        "checks": [
            "knapsack([1,3,4,5], [1,4,5,7], 7) == 9", "knapsack([], [], 10) == 0",
            "knapsack([5], [10], 4) == 0", "knapsack([2,2,2], [3,3,3], 4) == 6",
            "knapsack([0,1], [5,1], 0) == 5",
        ],
    },
    {
        "id": "word_break_all", "split": "holdout", "entry_point": "word_break_all",
        "prompt": "Write `word_break_all(s: str, words: list[str]) -> list[str]` returning every way to segment s into a "
                  "space-separated sequence of dictionary words (words may be reused), sorted lexicographically. "
                  "Return [] if impossible; return [''] if s is empty.",
        "checks": [
            "word_break_all('catsanddog', ['cat','cats','and','sand','dog']) == ['cat sand dog', 'cats and dog']",
            "word_break_all('pineapplepenapple', ['apple','pen','applepen','pine','pineapple']) == "
            "['pine apple pen apple', 'pine applepen apple', 'pineapple pen apple']",
            "word_break_all('catsandog', ['cats','dog','sand','and','cat']) == []",
            "word_break_all('', ['a']) == ['']",
            "word_break_all('aaa', ['a','aa']) == ['a a a', 'a aa', 'aa a']",
        ],
    },
    {
        "id": "next_permutation", "split": "holdout", "entry_point": "next_permutation",
        "prompt": "Write `next_permutation(nums: list[int]) -> list[int]` returning (as a new list) the next "
                  "lexicographically greater permutation of nums; if none exists return the lowest (sorted ascending). "
                  "Duplicates may occur.",
        "checks": [
            "next_permutation([1,2,3]) == [1,3,2]", "next_permutation([3,2,1]) == [1,2,3]",
            "next_permutation([1,1,5]) == [1,5,1]", "next_permutation([1,5,1]) == [5,1,1]",
            "next_permutation([]) == []", "next_permutation([2,3,1,3,3]) == [2,3,3,1,3]",
        ],
    },
    {
        "id": "longest_valid_parens", "split": "holdout", "entry_point": "longest_valid_parens",
        "prompt": "Write `longest_valid_parens(s: str) -> int` returning the length of the longest contiguous substring "
                  "of s that is a well-formed parentheses string. s contains only '(' and ')'.",
        "checks": [
            "longest_valid_parens('(()') == 2", "longest_valid_parens(')()())') == 4",
            "longest_valid_parens('') == 0", "longest_valid_parens('()(())') == 6",
            "longest_valid_parens('())(())') == 4", "longest_valid_parens('((((') == 0",
        ],
    },
    {
        "id": "islands", "split": "holdout", "entry_point": "islands",
        "prompt": "Write `islands(grid: list[str]) -> tuple[int, int]`. grid rows are strings of '1' (land) and '0' "
                  "(water). Islands are 4-connected land regions. Return (number of islands, size of the largest "
                  "island); (0, 0) if there is no land. The grid may be empty.",
        "checks": [
            "islands(['11000','11000','00100','00011']) == (3, 4)", "islands([]) == (0, 0)",
            "islands(['000']) == (0, 0)", "islands(['101','010','101']) == (5, 1)",
            "islands(['111','101','111']) == (1, 8)",
        ],
    },
    {
        "id": "kth_pair_distance", "split": "holdout", "entry_point": "kth_pair_distance",
        "prompt": "Write `kth_pair_distance(nums: list[int], k: int) -> int` returning the k-th smallest (1-based) "
                  "absolute difference among all pairs (i < j). It must run fast for len(nums) up to 20000.",
        "checks": [
            "kth_pair_distance([1,3,1], 1) == 0", "kth_pair_distance([1,1,1], 2) == 0",
            "kth_pair_distance([1,6,1], 3) == 5", "kth_pair_distance([9,1,4,7], 4) == 5",
            "kth_pair_distance(list(range(20000)), 199990000) == 19999",
        ],
    },
    {
        "id": "justify", "split": "holdout", "entry_point": "justify",
        "prompt": "Write `justify(words: list[str], width: int) -> list[str]` (every word length <= width). Greedily "
                  "pack words into lines. Every line except the last is fully justified to exactly `width` "
                  "characters: spaces are distributed as evenly as possible between words, with extra spaces going "
                  "to the leftmost gaps; a line with a single word is left-justified and padded with spaces. The last "
                  "line is left-justified with single spaces and padded with spaces to width.",
        "checks": [
            "justify(['This','is','an','example','of','text','justification.'], 16) == "
            "['This    is    an', 'example  of text', 'justification.  ']",
            "justify(['What','must','be','acknowledgment','shall','be'], 16) == "
            "['What   must   be', 'acknowledgment  ', 'shall be        ']",
            "justify(['a'], 3) == ['a  ']",
            "justify(['ab','c','d'], 4) == ['ab c', 'd   ']",
        ],
    },
    {
        "id": "parse_duration", "split": "holdout", "entry_point": "parse_duration",
        "prompt": "Write `parse_duration(s: str) -> int` converting strings like '1h30m15s' or '2d4s' to seconds. Units "
                  "are d, h, m, s; each must appear at most once, in that order (d before h before m before s), each "
                  "preceded by a non-negative integer. Raise ValueError for an empty string or any other malformed "
                  "input (wrong order, repeated unit, missing number, unknown unit, spaces).",
        "checks": [
            "parse_duration('1h30m15s') == 5415", "parse_duration('2d4s') == 172804",
            "parse_duration('0s') == 0", "parse_duration('90m') == 5400",
            "_raises(ValueError, parse_duration, '')", "_raises(ValueError, parse_duration, '5m1h')",
            "_raises(ValueError, parse_duration, '1h1h')", "_raises(ValueError, parse_duration, 'h')",
            "_raises(ValueError, parse_duration, '3x')", "_raises(ValueError, parse_duration, '1h 2m')",
            "_raises(ValueError, parse_duration, '15')",
        ],
    },
    {
        "id": "top_k_suggestions", "split": "holdout", "entry_point": "top_k_suggestions",
        "prompt": "Write `top_k_suggestions(freq: dict[str, int], prefix: str, k: int) -> list[str]` returning up to k "
                  "words from freq that start with prefix, ordered by frequency descending, then alphabetically.",
        "checks": [
            "top_k_suggestions({'apple':5,'app':5,'apply':2,'bat':9}, 'ap', 2) == ['app','apple']",
            "top_k_suggestions({'a':1}, 'b', 3) == []",
            "top_k_suggestions({'x':1,'xy':3,'xyz':2}, '', 5) == ['xy','xyz','x']",
            "top_k_suggestions({'aa':1,'ab':1}, 'a', 0) == []",
        ],
    },
    {
        "id": "rotate_cw", "split": "holdout", "entry_point": "rotate_cw",
        "prompt": "Write `rotate_cw(matrix: list[list[int]]) -> list[list[int]]` returning a new matrix that is the "
                  "input rotated 90 degrees clockwise. The input may be rectangular; return [] for an empty matrix.",
        "checks": [
            "rotate_cw([[1,2],[3,4]]) == [[3,1],[4,2]]", "rotate_cw([[1,2,3]]) == [[1],[2],[3]]",
            "rotate_cw([[1],[2],[3]]) == [[3,2,1]]", "rotate_cw([]) == []",
            "rotate_cw([[1,2,3],[4,5,6]]) == [[4,1],[5,2],[6,3]]",
        ],
    },
    {
        "id": "sum_fractions", "split": "holdout", "entry_point": "sum_fractions",
        "prompt": "Write `sum_fractions(fracs: list[str]) -> str`. Each item is 'a/b' with integers a and b (b != 0, "
                  "either may be negative). Return the sum in lowest terms as 'p/q' with q > 0; zero is '0/1'; an "
                  "empty list sums to '0/1'.",
        "checks": [
            "sum_fractions(['1/2','1/3']) == '5/6'", "sum_fractions(['1/2','-1/2']) == '0/1'",
            "sum_fractions([]) == '0/1'", "sum_fractions(['3/-6']) == '-1/2'",
            "sum_fractions(['-2/-4','1/4']) == '3/4'", "sum_fractions(['4/2']) == '2/1'",
        ],
    },

    # ---------------------------------------------------------------- audit
    # Measurement only: never used for selection or promotion, never shown to
    # any genome, and excluded from the GUI's per-task view.
    {
        "id": "roman_to_int", "split": "audit", "entry_point": "roman_to_int",
        "prompt": "Write `roman_to_int(s: str) -> int` converting a Roman numeral to an integer. Accept only the "
                  "canonical standard form for 1..3999 (exactly what a correct int-to-Roman converter using subtractive "
                  "notation would produce, uppercase). Raise ValueError for anything else, e.g. 'IIII', 'IC', 'VV', "
                  "'', 'mcm'.",
        "checks": [
            "roman_to_int('MCMXCIV') == 1994", "roman_to_int('III') == 3", "roman_to_int('MMMCMXCIX') == 3999",
            "roman_to_int('XLII') == 42", "_raises(ValueError, roman_to_int, 'IIII')",
            "_raises(ValueError, roman_to_int, 'IC')", "_raises(ValueError, roman_to_int, 'VV')",
            "_raises(ValueError, roman_to_int, '')", "_raises(ValueError, roman_to_int, 'mcm')",
        ],
    },
    {
        "id": "group_anagrams", "split": "audit", "entry_point": "group_anagrams",
        "prompt": "Write `group_anagrams(words: list[str]) -> list[list[str]]`. Group words that are anagrams of each "
                  "other (same letters with the same multiplicity, case-sensitive). Sort each group, keep duplicates, "
                  "and sort the groups by their first element.",
        "checks": [
            "group_anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'], ['bat'], ['nat','tan']]",
            "group_anagrams([]) == []", "group_anagrams(['a','a']) == [['a','a']]",
            "group_anagrams(['ab','Ba','ba']) == [['Ba'], ['ab','ba']]",
            "group_anagrams(['', '']) == [['', '']]",
        ],
    },
    {
        "id": "max_subarray", "split": "audit", "entry_point": "max_subarray",
        "prompt": "Write `max_subarray(nums: list[int]) -> tuple[int, int, int]` returning (sum, start, end) of the "
                  "non-empty contiguous subarray nums[start:end+1] with the largest sum. Break ties by the smallest "
                  "start, then by the smallest end. nums is non-empty.",
        "checks": [
            "max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == (6, 3, 6)", "max_subarray([-3,-1,-2]) == (-1, 1, 1)",
            "max_subarray([5]) == (5, 0, 0)", "max_subarray([1,-1,1]) == (1, 0, 0)",
            "max_subarray([0,0,0]) == (0, 0, 0)", "max_subarray([2,-5,2]) == (2, 0, 0)",
        ],
    },
    {
        "id": "flatten_json", "split": "audit", "entry_point": "flatten_json",
        "prompt": "Write `flatten_json(obj: dict) -> dict` flattening nested dicts and lists into a single dict whose "
                  "keys are dot-joined paths; list elements use their index as the path segment. Empty dicts and "
                  "empty lists are kept as leaf values. Example: {'a': {'b': 1}, 'c': [1, {'d': 2}]} -> "
                  "{'a.b': 1, 'c.0': 1, 'c.1.d': 2}.",
        "checks": [
            "flatten_json({'a': {'b': 1}, 'c': [1, {'d': 2}]}) == {'a.b': 1, 'c.0': 1, 'c.1.d': 2}",
            "flatten_json({}) == {}", "flatten_json({'x': {}, 'y': []}) == {'x': {}, 'y': []}",
            "flatten_json({'a': [[1, 2], [3]]}) == {'a.0.0': 1, 'a.0.1': 2, 'a.1.0': 3}",
            "flatten_json({'n': None, 's': 'str'}) == {'n': None, 's': 'str'}",
        ],
    },
    {
        "id": "compress_ranges", "split": "audit", "entry_point": "compress_ranges",
        "prompt": "Write `compress_ranges(nums: list[int]) -> str`. Deduplicate and sort the integers, then describe "
                  "them as comma-separated items where runs of 3 or more consecutive integers become 'a-b' and "
                  "shorter runs are listed individually. Negative bounds are written as-is, e.g. '-3--1'. Return '' "
                  "for an empty list.",
        "checks": [
            "compress_ranges([1,2,3,5,7,8]) == '1-3,5,7,8'", "compress_ranges([]) == ''",
            "compress_ranges([3,1,2,2]) == '1-3'", "compress_ranges([-3,-2,-1,1]) == '-3--1,1'",
            "compress_ranges([10]) == '10'", "compress_ranges([1,2,4,5,6]) == '1,2,4-6'",
        ],
    },
    {
        "id": "word_ladder", "split": "audit", "entry_point": "word_ladder",
        "prompt": "Write `word_ladder(begin: str, end: str, words: list[str]) -> int` returning the number of words in "
                  "the shortest sequence from begin to end where each step changes exactly one letter and every word "
                  "after begin is in `words` (all words have equal length). Return 0 if impossible; return 1 if "
                  "begin == end.",
        "checks": [
            "word_ladder('hit', 'cog', ['hot','dot','dog','lot','log','cog']) == 5",
            "word_ladder('hit', 'cog', ['hot','dot','dog','lot','log']) == 0",
            "word_ladder('a', 'c', ['a','b','c']) == 2", "word_ladder('abc', 'abc', []) == 1",
            "word_ladder('ab', 'cd', ['ad','cd']) == 3",
        ],
    },
    {
        "id": "grid_paths", "split": "audit", "entry_point": "grid_paths",
        "prompt": "Write `grid_paths(grid: list[str]) -> int` counting paths from the top-left to the bottom-right "
                  "cell moving only right or down, never entering '#' cells ('.' is open). Return 0 if the start or "
                  "end is blocked or the grid is empty. Results can be very large; return the exact integer.",
        "checks": [
            "grid_paths(['...','.#.','...']) == 2", "grid_paths(['#']) == 0", "grid_paths(['.']) == 1",
            "grid_paths([]) == 0", "grid_paths(['..','.#']) == 0",
            "grid_paths(['.' * 30] * 30) == 30067266499541040",
        ],
    },
    {
        "id": "eval_rpn", "split": "audit", "entry_point": "eval_rpn",
        "prompt": "Write `eval_rpn(tokens: list[str]) -> int` evaluating a Reverse Polish Notation expression of "
                  "integers (possibly negative, like '-3') and the operators + - * /, where / truncates toward zero. "
                  "Raise ValueError if the expression is malformed (an operator without two operands, leftover "
                  "operands, an empty list, or an unknown token).",
        "checks": [
            "eval_rpn(['2','1','+','3','*']) == 9", "eval_rpn(['4','13','5','/','+']) == 6",
            "eval_rpn(['-7','2','/']) == -3", "eval_rpn(['42']) == 42",
            "_raises(ValueError, eval_rpn, [])", "_raises(ValueError, eval_rpn, ['1','+'])",
            "_raises(ValueError, eval_rpn, ['1','2'])", "_raises(ValueError, eval_rpn, ['1','2','^'])",
        ],
    },
    {
        "id": "min_rooms", "split": "audit", "entry_point": "min_rooms",
        "prompt": "Write `min_rooms(meetings: list[tuple[int, int]]) -> int` returning the minimum number of rooms "
                  "needed so that no two meetings in the same room overlap. Meetings are half-open [start, end), so "
                  "one ending at 10 and another starting at 10 can share a room. Zero-length meetings (start == end) "
                  "need no room.",
        "checks": [
            "min_rooms([(0,30),(5,10),(15,20)]) == 2", "min_rooms([(7,10),(2,4)]) == 1",
            "min_rooms([]) == 0", "min_rooms([(1,5),(5,10)]) == 1",
            "min_rooms([(1,10),(2,10),(3,10)]) == 3", "min_rooms([(3,3),(3,3)]) == 0",
        ],
    },
    {
        "id": "base_convert", "split": "audit", "entry_point": "base_convert",
        "prompt": "Write `base_convert(s: str, from_base: int, to_base: int) -> str` converting an integer written in "
                  "from_base (2..36, digits 0-9 then letters, case-insensitive, optional leading '-') to to_base, "
                  "using uppercase letters, no leading zeros, and '0' for zero. Raise ValueError for invalid digits, "
                  "empty input or bases outside 2..36. Do not use int(s, base).",
        "checks": [
            "base_convert('ff', 16, 2) == '11111111'", "base_convert('-255', 10, 16) == '-FF'",
            "base_convert('0', 10, 2) == '0'", "base_convert('zz', 36, 10) == '1295'",
            "base_convert('-0', 10, 7) == '0'", "_raises(ValueError, base_convert, '12', 2, 10)",
            "_raises(ValueError, base_convert, '', 10, 2)", "_raises(ValueError, base_convert, '1', 1, 10)",
        ],
    },
    {
        "id": "scs_length", "split": "audit", "entry_point": "scs_length",
        "prompt": "Write `scs_length(a: str, b: str) -> int` returning the length of the shortest string that has both "
                  "a and b as (not necessarily contiguous) subsequences.",
        "checks": [
            "scs_length('abac', 'cab') == 5", "scs_length('', 'abc') == 3", "scs_length('abc', 'abc') == 3",
            "scs_length('abc', 'def') == 6", "scs_length('a' * 300, 'a' * 200 + 'b') == 301",
        ],
    },
    {
        "id": "cron_match", "split": "audit", "entry_point": "cron_match",
        "prompt": "Write `cron_match(expr: str, minute: int, hour: int) -> bool`. expr has two space-separated "
                  "fields: minute (0-59) and hour (0-23). Each field is a comma-separated list of items, each one of "
                  "'*', 'N', 'A-B' (inclusive), '*/S' or 'A-B/S' (every S-th value starting at A, or at the field "
                  "minimum for '*'). Return True iff both fields match.",
        "checks": [
            "cron_match('*/15 9-17', 30, 12) is True", "cron_match('*/15 9-17', 31, 12) is False",
            "cron_match('0 0', 0, 0) is True", "cron_match('5,10-12 *', 11, 23) is True",
            "cron_match('10-20/5 *', 15, 1) is True", "cron_match('10-20/5 *', 16, 1) is False",
            "cron_match('* 22-23', 0, 21) is False", "cron_match('*/7 */5', 56, 20) is True",
        ],
    },
]
