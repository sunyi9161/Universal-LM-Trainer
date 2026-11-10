#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenizer tester — 交互式 BPE 分词器编码/解码工具

只依赖 `tokenizers`，不加载模型，不占显存，用来快速检查：

  - 一段文本被切成了哪些 token、各自的 id、以及每个 token 覆盖的原始字符区间
  - encode -> decode 是否可逆（不一致时指出第一个出错的字符位置）
  - 一串 id 会解码成什么、每个 id 对应哪个 token
  - 整个文件（纯文本或 jsonl）的 token/字符比、<unk> 数量、不可逆行数

用法
----
    python tok_test.py                          # 交互模式 (REPL)
    python tok_test.py --text "今天天气不错"      # 单次编码
    python tok_test.py --text "..." --span       # 带字符区间
    python tok_test.py --ids "2642 3532 5211"    # 单次解码
    python tok_test.py --file corpus.jsonl       # 文件统计
    python tok_test.py --selftest                # 内置样例自检
    echo "你好" | python tok_test.py             # 管道: 每行输出一串 id

交互模式命令（以 `:` 开头，其他输入一律当作文本编码）
--------------------------------------------------------
    :help / :h            帮助
    :decode <ids> / :d    id -> token -> 文本
    :file <path> / :f     文件统计
    :info                 词表大小、特殊 token、pre_tokenizer / decoder
    :test / :t            跑内置样例自检
    :limit <n> / :l       表格最多显示多少个 token (0 = 全部)
    :span / :s            切换是否显示字符区间
    :special              切换是否在首尾手动包 <bos>/<eos>
    :quit / :q            退出

默认分词器路径是脚本同目录的 tokenizer.json，可用 --tokenizer 或环境变量
TOK_TEST_TOKENIZER 覆盖。
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover
    sys.exit('缺少依赖: pip install tokenizers')

DEFAULT_TOKENIZER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json')
SPECIALS = ['<pad>', '<unk>', '<bos>', '<eos>']

SAMPLES = [
    '今天中国的首都是哪里？',
    'Hello, world!',
    '混合 text 与中文 and 数字 123456.',
    '多行\n文本\t带制表符\r\n',
    '  前后都有空格的文本  ',
    '<unk> <bos> <eos> 特殊 token',
    'emoji 😀🎉 和全角标点，。！？',
    '',
]


# --------------------------------------------------------------------------
# 终端显示
# --------------------------------------------------------------------------
class C:
    enabled = sys.stdout.isatty() and os.environ.get('NO_COLOR') is None

    @classmethod
    def wrap(cls, s, code):
        return f'\033[{code}m{s}\033[0m' if cls.enabled else s

    @classmethod
    def dim(cls, s):
        return cls.wrap(s, '2')

    @classmethod
    def bold(cls, s):
        return cls.wrap(s, '1')

    @classmethod
    def cyan(cls, s):
        return cls.wrap(s, '36')

    @classmethod
    def green(cls, s):
        return cls.wrap(s, '32')

    @classmethod
    def red(cls, s):
        return cls.wrap(s, '31')

    @classmethod
    def yellow(cls, s):
        return cls.wrap(s, '33')


def dwidth(s):
    """终端显示宽度（中日韩全角按 2 列算），用于表格对齐。"""
    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
    return w


def pad(s, width, right=False):
    fill = ' ' * max(0, width - dwidth(s))
    return fill + s if right else s + fill


def visible(token):
    """token 的可读形式：保留原文，但把空白/不可见字符转义，避免撑乱表格。"""
    return (token.replace('\\', '\\\\')
                 .replace('\n', '\\n')
                 .replace('\r', '\\r')
                 .replace('\t', '\\t'))


def first_diff(a, b):
    """返回第一个不同字符的下标（越界时取较短串长度）。"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# --------------------------------------------------------------------------
# 核心功能
# --------------------------------------------------------------------------
def load_tokenizer(path):
    if not os.path.exists(path):
        sys.exit(f'找不到分词器文件: {path}\n用 --tokenizer 指定路径，或设置 TOK_TEST_TOKENIZER。')
    try:
        return Tokenizer.from_file(path)
    except Exception as e:
        sys.exit(f'读取分词器失败: {e}')


def encode_one(tok, text):
    """encode 并返回 (ids, tokens, offsets)。tokenizers 的 offsets 是原文区间。"""
    enc = tok.encode(text)
    return enc.ids, enc.tokens, enc.offsets


def show_encoding(tok, text, limit=40, show_span=False, add_special=False):
    if not text:
        print(C.dim('(空输入)'))
        return
    enc = tok.encode(text, add_special_tokens=add_special)
    ids, tokens, offsets = list(enc.ids), list(enc.tokens), list(enc.offsets)

    # 这个 tokenizer 没有 post_processor，encode 自己不会加特殊 token；
    # 打开 :special 时手动在首尾包 <bos>/<eos>，方便看模型实际吃到的序列。
    if add_special and tok.post_processor is None:
        bos = tok.token_to_id('<bos>')
        eos = tok.token_to_id('<eos>')
        if bos is not None:
            ids.insert(0, bos)
            tokens.insert(0, '<bos>')
            offsets.insert(0, (0, 0))
        if eos is not None:
            ids.append(eos)
            tokens.append('<eos>')
            offsets.append((len(text), len(text)))

    n_show = len(ids) if limit <= 0 else min(limit, len(ids))
    id_w = max([len('id')] + [len(str(i)) for i in ids[:n_show]])
    tok_w = max([len('token')] + [dwidth(visible(t)) for t in tokens[:n_show]])

    head = f"{pad('#', 4, True)}  {pad('id', id_w, True)}  {pad('token', tok_w)}"
    if show_span:
        head += '  span'
    print(C.dim(head))
    print(C.dim('-' * max(20, dwidth(head))))

    for i in range(n_show):
        line = f"{pad(str(i), 4, True)}  {pad(str(ids[i]), id_w, True)}  " \
               f"{pad(visible(tokens[i]), tok_w)}"
        if show_span:
            a, b = offsets[i]
            line += f"  [{a},{b})"
        print(line)
    if n_show < len(ids):
        print(C.dim(f'... 还有 {len(ids) - n_show} 个 token ( :limit 0 显示全部 )'))

    n_chars = len(text)
    ratio = len(ids) / n_chars if n_chars else 0.0
    print()
    print(f'字符数 {C.bold(str(n_chars))}   token 数 {C.bold(str(len(ids)))}   '
          f'token/字符 {ratio:.3f}')

    # 可逆性检查（加了特殊 token 时解码要跳过它们）
    dec = tok.decode(ids, skip_special_tokens=add_special)
    if dec == text:
        print(C.green('roundtrip: 完全一致 ✓'))
    else:
        i = first_diff(dec, text)
        print(C.red(f'roundtrip: 不一致 ✗  (第 {i} 个字符起)'))
        print(f'  原文: {text[max(0, i - 20):i + 20]!r}')
        print(f'  解码: {dec[max(0, i - 20):i + 20]!r}')

    # 特殊 token 提示
    sp = [tok.id_to_token(t) for t in ids if tok.id_to_token(t) in SPECIALS]
    if sp:
        print(C.yellow(f'包含特殊 token: {dict(Counter(sp))}'))


def show_decoding(tok, ids):
    if not ids:
        print(C.yellow('没有给出任何 id'))
        return
    vocab = tok.get_vocab_size()
    oob = [i for i in ids if i < 0 or i >= vocab]
    if oob:
        print(C.red(f'越界 id (词表大小 {vocab}): {oob}'))

    rows = []
    for i in ids:
        t = tok.id_to_token(i)
        rows.append((i, t))
    id_w = max(len(str(i)) for i, _ in rows)
    tok_w = max(len('token') if t is None else dwidth(visible(t)) for _, t in rows)
    tok_w = max(tok_w, len('token'))

    print(C.dim(f"{pad('#', 4, True)}  {pad('id', id_w, True)}  {pad('token', tok_w)}"))
    print(C.dim('-' * max(20, 4 + 2 + id_w + 2 + tok_w)))
    for n, (i, t) in enumerate(rows):
        if t is None:
            cell = C.red(pad('(越界)', tok_w))
        elif t in SPECIALS:
            cell = C.yellow(pad(visible(t), tok_w))
        else:
            cell = pad(visible(t), tok_w)
        print(f"{pad(str(n), 4, True)}  {pad(str(i), id_w, True)}  {cell}")

    print()
    if oob:
        print(C.yellow(f'解码时忽略 {len(oob)} 个越界 id'))
    valid = [i for i in ids if 0 <= i < vocab]
    dec_all = tok.decode(valid, skip_special_tokens=False)
    print('解码(保留特殊 token): ' + repr(dec_all))
    print('解码(跳过特殊 token): ' + repr(tok.decode(valid, skip_special_tokens=True)))
    if valid and not dec_all:
        print(C.yellow('提示: 非空 id 解出空串 —— Metaspace 会吞掉孤立的 "▁" token。'))


def parse_ids(text):
    out = []
    for part in re.split(r'[\s,]+', text.strip()):
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            print(C.red(f'忽略非数字: {part!r}'))
    return out


def file_stats(tok, path):
    if not os.path.exists(path):
        print(C.red(f'文件不存在: {path}'))
        return

    n_lines = n_chars = n_tokens = n_unk = n_bad = 0
    freq = Counter()
    t0 = os.times().elapsed

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            # 支持本项目的 jsonl: 优先取 "text" / "content" 字段
            if line.lstrip().startswith('{'):
                try:
                    obj = json.loads(line)
                    line = obj.get('text', obj.get('content', line))
                except Exception:
                    pass
            ids = tok.encode(line).ids
            dec = tok.decode(ids, skip_special_tokens=False)
            n_lines += 1
            n_chars += len(line)
            n_tokens += len(ids)
            n_unk += sum(1 for i in ids if tok.id_to_token(i) == '<unk>')
            if dec != line:
                n_bad += 1
            freq.update(ids)

    dt = os.times().elapsed - t0
    ratio = n_tokens / n_chars if n_chars else 0.0
    print()
    print(f'文件: {path}')
    print(f'  行数        {n_lines:,}')
    print(f'  字符数      {n_chars:,}')
    print(f'  token 数    {n_tokens:,}')
    print(f'  token/字符  {ratio:.3f}')
    print(f'  <unk> 数    {n_unk:,}')
    print(f'  不可逆行数  {n_bad:,}  ("encode->decode != 原文")')
    print(f'  耗时        {dt:.2f}s  ' + (f'({n_chars / dt:,.0f} 字符/s)' if dt > 1e-6 else '(太快, 无法计时)'))

    if freq:
        print('  高频 token (前 10):')
        for tid, cnt in freq.most_common(10):
            print(f'    {tid:>6}  {pad(visible(tok.id_to_token(tid) or "?"), 12)} {cnt:,}')


def show_info(tok):
    vocab = tok.get_vocab_size()
    print(f'词表大小   {vocab:,}')
    print(f'pre_tokenizer  {tok.pre_tokenizer}')
    print(f'decoder        {tok.decoder}')
    print(f'normalizer     {tok.normalizer}')
    print(f'post_processor {tok.post_processor}')
    print(f'padding        {tok.padding}')
    print(f'truncation     {tok.truncation}')
    print('特殊 token:')
    for s in SPECIALS:
        tid = tok.token_to_id(s)
        print(f'  {s:<8} {"未定义" if tid is None else tid}')
    missing = [s for s in SPECIALS if tok.token_to_id(s) is None]
    if missing:
        print(C.yellow(f'⚠️ 缺少特殊 token: {missing}'))


def selftest(tok):
    print(C.bold('内置样例自检'))
    print(C.dim('-' * 62))
    ok = warn = bad = 0
    for text in SAMPLES:
        enc = tok.encode(text)
        dec = tok.decode(enc.ids, skip_special_tokens=False)
        same = dec == text
        # Metaspace(prepend_scheme=always) 会把开头的空格并进 token，解码时无法
        # 还原；只有"去掉首尾空白后仍不一致"才算真正失败。
        soft = (not same) and dec.strip() == text.strip()
        ok += same
        warn += soft
        bad += not (same or soft)
        if same:
            mark = C.green('✓')
        elif soft:
            mark = C.yellow('~')
        else:
            mark = C.red('✗')
        print(f'{mark} {len(enc.ids):>4} tok  {visible(text)[:36]}')
        if not same:
            i = first_diff(dec, text)
            note = '首尾空白丢失(分词器固有行为)' if soft else '真实差异'
            print(f'     {note} @{i}: 原文={text[i:i + 12]!r} 解码={dec[i:i + 12]!r}')
    print(C.dim('-' * 62))
    print(f'完全可逆 {ok}  空白丢失 {warn}  失败 {bad}')
    return bad == 0


# --------------------------------------------------------------------------
# 交互模式
# --------------------------------------------------------------------------
HELP = """
命令 (以 : 开头)
  :help / :h           显示本帮助
  :decode <ids> / :d   id -> token -> 文本，例如  :d 2642 3532 5211
  :file <path> / :f    统计文件 (纯文本或 jsonl)
  :info                词表大小 / 特殊 token / 前后处理配置
  :test / :t           跑内置样例自检
  :limit <n> / :l      表格最多显示 n 个 token (0 = 全部)
  :span / :s           切换是否显示字符区间
  :special             切换是否在首尾手动包 <bos>/<eos>
  :quit / :q           退出

其他任何输入都会被当作要编码的文本。
""".strip()


def repl(tok, args):
    print(C.bold('Tokenizer Tester'))
    print(f'  分词器: {args.tokenizer}')
    print(f'  词表:   {tok.get_vocab_size():,}')
    print(C.dim('输入文本直接编码；:help 查看命令；Ctrl-C / Ctrl-D 退出'))
    if tok.post_processor is None:
        print(C.dim('  提示: 该 tokenizer 无 post_processor，:special 会手动在首尾包 <bos>/<eos>'))
    print()

    limit, show_span, add_special = args.limit, args.span, args.special

    while True:
        try:
            line = input(C.cyan('tok> '))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            continue

        if line.startswith(':'):
            name, _, rest = line[1:].strip().partition(' ')
            name = name.lower()
            rest = rest.strip()
            if name in ('q', 'quit', 'exit'):
                break
            elif name in ('h', 'help'):
                print(HELP)
            elif name in ('d', 'decode'):
                show_decoding(tok, parse_ids(rest))
            elif name in ('f', 'file'):
                if rest:
                    file_stats(tok, os.path.expanduser(rest))
                else:
                    print(C.yellow('用法: :file <path>'))
            elif name in ('i', 'info'):
                show_info(tok)
            elif name in ('t', 'test'):
                selftest(tok)
            elif name in ('l', 'limit'):
                try:
                    limit = max(0, int(rest))
                    print(C.dim(f'limit = {limit}' + (' (全部)' if limit == 0 else '')))
                except ValueError:
                    print(C.yellow('用法: :limit <n>'))
            elif name in ('s', 'span'):
                show_span = not show_span
                print(C.dim(f'span = {show_span}'))
            elif name == 'special':
                add_special = not add_special
                print(C.dim(f'add_special = {add_special}'))
            else:
                print(C.yellow(f'未知命令 :{name}，输入 :help 查看帮助'))
            print()
            continue

        show_encoding(tok, line, limit=limit, show_span=show_span,
                      add_special=add_special)
        print()


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
USAGE = """\
示例:
  python %(prog)s                         # 交互模式 (REPL)
  python %(prog)s --text "今天天气不错"     # 单次编码
  python %(prog)s --text "..." --span      # 带字符区间
  python %(prog)s --ids "2642 3532 5211"   # 单次解码
  python %(prog)s --file corpus.jsonl      # 文件统计
  python %(prog)s --selftest               # 内置样例自检
  echo "你好" | python %(prog)s            # 管道: 每行输出一串 id
"""


def main():
    p = argparse.ArgumentParser(
        description='交互式 tokenizer 编码/解码测试器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE,
    )
    p.add_argument('--tokenizer', default=os.environ.get('TOK_TEST_TOKENIZER', DEFAULT_TOKENIZER),
                   help='tokenizer.json 路径 (默认: 脚本同目录)')
    p.add_argument('--text', help='编码这段文本后退出')
    p.add_argument('--ids', help='解码这些 id 后退出 (空格或逗号分隔)')
    p.add_argument('--file', help='统计文件后退出 (纯文本或 jsonl)')
    p.add_argument('--info', action='store_true', help='打印词表/特殊 token 信息后退出')
    p.add_argument('--selftest', action='store_true', help='跑内置样例自检后退出')
    p.add_argument('--limit', type=int, default=40, help='表格最多显示多少 token，0=全部 (默认 40)')
    p.add_argument('--span', action='store_true', help='显示每个 token 的字符区间')
    p.add_argument('--special', action='store_true', help='在首尾包 <bos>/<eos>')
    p.add_argument('--no-color', action='store_true', help='关闭颜色')
    args = p.parse_args()

    if args.no_color:
        C.enabled = False
    args.tokenizer = os.path.abspath(os.path.expanduser(args.tokenizer))
    tok = load_tokenizer(args.tokenizer)

    if args.selftest:
        sys.exit(0 if selftest(tok) else 1)
    if args.info:
        show_info(tok)
        return
    if args.ids is not None:
        show_decoding(tok, parse_ids(args.ids))
        return
    if args.file:
        file_stats(tok, os.path.expanduser(args.file))
        return
    if args.text is not None:
        show_encoding(tok, args.text, limit=args.limit, show_span=args.span,
                      add_special=args.special)
        return

    if not sys.stdin.isatty():
        # 管道模式: 一行输入对应一行 id，空行也保留（保持行号一一对应）
        for line in sys.stdin:
            line = line.rstrip('\n')
            print(' '.join(map(str, tok.encode(line).ids)))
        return

    repl(tok, args)


if __name__ == '__main__':
    main()
