#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 flat_u16 训练缓存里“文本内字面 <bos>/<eos>”被编码成 id 2/3 的样本。

背景
----
旧分词器把 <bos>/<eos> 放在 added_tokens 里，文本里字面出现这两个字符串时
会被直接匹配成特殊 token id 2/3。现在 train.py/inf.py 的 BPETokenizer 已经
改成用一个“去掉这两个 added token”的副本来编码正文，因此新编码不会再有
这个问题。但已经写好的 16GB 缓存里仍有极少量遗留 id 2/3（全量扫描为 53 个，
分布在 30 个样本、20 个桶里）。

做法
----
只重写受影响的桶：
  1. 用“普通编码器”decode 原始样本（id 2/3 会显示成字面 <bos>/<eos>）；
  2. 再用普通编码器 encode 回 token；
  3. 若新长度超过该桶上限，直接丢弃该样本（不截断）。
注意：丢弃样本会改变桶内样本数，checkpoint 的 counts_md5 可能不再匹配，
修复后需要从头训练或重新建立恢复点。
重写前会把旧文件改名为 *.orig，可用 --restore 回滚。

用法
----
  python repair_cache_special_tokens.py            # 干跑，只报告
  python repair_cache_special_tokens.py --apply    # 实际修复
  python repair_cache_special_tokens.py --restore  # 回滚
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.environ.setdefault('FA_V100_DISABLE', '1')
os.environ.setdefault('UNIVERSAL_LM_COMPILE', '0')

import train  # noqa: E402

# 默认从 <data_dir>/bucketed_cache_streaming_<max_seq_len>/<cache_id>/flat_u16 读取；
# 请用 --flat 指向你的实际缓存目录。
DEFAULT_FLAT = None


def _scan_positions(toks, chunk=50_000_000):
    positions = []
    for s in range(0, len(toks), chunk):
        ch = np.asarray(toks[s:s + chunk])
        m = (ch == 2) | (ch == 3)
        if m.any():
            positions.extend((s + np.nonzero(m)[0]).tolist())
    return positions


def _affected_samples(offs, positions):
    if not positions:
        return []
    pos = np.asarray(positions, dtype=np.int64)
    return sorted(set((np.searchsorted(offs[:-1], pos, side='right') - 1).tolist()))


def _rebuild(toks_path, offs_path, boundary, plain):
    toks = np.memmap(toks_path, dtype=np.uint16, mode='r')
    offs = np.load(offs_path).astype(np.int64)
    n = len(offs) - 1

    positions = _scan_positions(toks)
    sis = _affected_samples(offs, positions)
    fixed = {}
    dropped = set()
    changes = []
    for si in sis:
        a, z = int(offs[si]), int(offs[si + 1])
        old = [int(x) for x in np.asarray(toks[a:z])]
        text = plain.decode(old)
        new = plain.encode(text).ids
        raw_len = len(new)
        if raw_len > boundary:
            # 与训练脚本一致：超长文档直接丢弃，不截断。
            dropped.add(si)
            changes.append((si, len(old), 0, raw_len))
            continue
        assert 2 not in new and 3 not in new, (si, len(new))
        fixed[si] = np.asarray(new, dtype=np.uint16)
        changes.append((si, len(old), len(new), raw_len))

    new_total = (len(toks)
                 - sum(int(offs[i + 1] - offs[i]) for i in dropped)
                 + sum(len(v) - int(offs[i + 1] - offs[i])
                       for i, v in fixed.items()))
    out = np.empty(new_total, dtype=np.uint16)
    new_offs = np.zeros(n - len(dropped) + 1, dtype=np.int64)
    pos = 0
    k = 0
    for i in range(n):
        if i in dropped:
            continue
        arr = fixed.get(i)
        if arr is None:
            a, z = int(offs[i]), int(offs[i + 1])
            arr = toks[a:z]
        out[pos:pos + len(arr)] = arr
        pos += len(arr)
        k += 1
        new_offs[k] = pos
    assert pos == new_total
    assert k == n - len(dropped)
    return out, new_offs, changes


def _atomic_write_tokens(path, arr):
    tmp = path + '.repair_tmp'
    with open(tmp, 'wb') as f:
        f.write(arr.tobytes())
        f.flush()
        os.fsync(f.fileno())
    return tmp


def _atomic_write_offsets(path, arr):
    tmp = path + '.repair_tmp'
    with open(tmp, 'wb') as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    return tmp


def _write_manifest(path, manifest):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _verify(flat, manifest, plain):
    ok = True
    for bstr, info in sorted(manifest['buckets'].items(), key=lambda kv: int(kv[0])):
        n = int(info['num_samples'])
        if n == 0:
            continue
        toks_path = os.path.join(flat, f'bucket_{bstr}.tokens.u16')
        offs_path = os.path.join(flat, f'bucket_{bstr}.offsets.npy')
        size = os.path.getsize(toks_path)
        if size != int(info['num_tokens']) * 2:
            print(f'❌ bucket {bstr}: 文件大小 {size} != num_tokens*2')
            ok = False
            continue
        offs = np.load(offs_path, mmap_mode='r')
        if len(offs) != n + 1 or int(offs[-1]) != int(info['num_tokens']):
            print(f'❌ bucket {bstr}: offsets 与 num_samples/num_tokens 不一致')
            ok = False
            continue
        toks = np.memmap(toks_path, dtype=np.uint16, mode='r')
        positions = _scan_positions(toks, chunk=10_000_000)
        if positions:
            print(f'❌ bucket {bstr}: 仍有 {len(positions)} 个特殊 id')
            ok = False
        over = 0
        for i in range(n):
            if int(offs[i + 1] - offs[i]) > int(bstr):
                over += 1
        if over:
            print(f'❌ bucket {bstr}: {over} 个样本超过桶上限')
            ok = False
    return ok


def restore(flat, manifest):
    restored = 0
    for path in sorted(list(_iter_backup_files(flat))):
        target = os.path.join(flat, os.path.basename(path)[:-len('.orig')])
        os.replace(path, target)
        restored += 1
    # 兼容上次中断时直接留在缓存目录里的 .orig
    for path in sorted(list(_iter_files(flat, '.orig'))):
        target = path[:-len('.orig')]
        os.replace(path, target)
        restored += 1
    bak = os.path.join(flat, 'manifest.json.bak')
    if os.path.exists(bak):
        os.replace(bak, os.path.join(flat, 'manifest.json'))
        print('✓ 已恢复 manifest.json')
    print(f'✓ 已恢复 {restored} 个文件')


def _iter_files(flat, suffix):
    for name in os.listdir(flat):
        if name.endswith(suffix):
            yield os.path.join(flat, name)


def _backup_dir(flat):
    return os.path.join(flat, 'repair_backup')


def _iter_backup_files(flat):
    bdir = _backup_dir(flat)
    if not os.path.isdir(bdir):
        return
    for name in sorted(os.listdir(bdir)):
        if name.endswith('.orig'):
            yield os.path.join(bdir, name)


def _archive_backups(flat):
    """把 *.orig 移到 repair_backup/，避免和正式缓存文件混在一起。"""
    bdir = _backup_dir(flat)
    os.makedirs(bdir, exist_ok=True)
    moved = 0
    for path in list(_iter_files(flat, '.orig')):
        os.replace(path, os.path.join(bdir, os.path.basename(path)))
        moved += 1
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat', default=DEFAULT_FLAT)
    ap.add_argument('--tokenizer', default=os.path.join(BASE, 'tokenizer.json'))
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()
    if not args.flat:
        ap.error('请用 --flat 指定 flat_u16 缓存目录（例如 cache/<cache_id>/flat_u16）')

    manifest_path = os.path.join(args.flat, 'manifest.json')
    manifest = json.load(open(manifest_path))

    if args.restore:
        restore(args.flat, manifest)
        return

    tok = train.BPETokenizer()
    tok.load(args.tokenizer)
    plain = tok._plain_tokenizer

    if list(_iter_files(args.flat, '.orig')):
        print('❌ 发现上次修复中断留下的 .orig 文件。先运行 --restore 回滚，再重新修复。')
        sys.exit(1)

    if args.apply:
        bak = manifest_path + '.bak'
        if not os.path.exists(bak):
            shutil.copy2(manifest_path, bak)
            print(f'✓ 已备份 manifest -> {os.path.basename(bak)}')

    total_changes = 0
    total_positions = 0
    for bstr, info in sorted(manifest['buckets'].items(), key=lambda kv: int(kv[0])):
        n = int(info['num_samples'])
        if n == 0:
            continue
        B = int(bstr)
        toks_path = os.path.join(args.flat, f'bucket_{B}.tokens.u16')
        offs_path = os.path.join(args.flat, f'bucket_{B}.offsets.npy')
        toks = np.memmap(toks_path, dtype=np.uint16, mode='r')
        positions = _scan_positions(toks)
        if not positions:
            continue
        offs = np.load(offs_path)
        sis = _affected_samples(offs, positions)
        print(f'bucket {B}: {len(positions)} 个特殊 id，{len(sis)} 个样本')
        for si in sis:
            a, z = int(offs[si]), int(offs[si + 1])
            print(f'  sample {si}: len {z - a}')
        total_positions += len(positions)
        total_changes += len(sis)

        if not args.apply:
            continue

        out, new_offs, changes = _rebuild(toks_path, offs_path, B, plain)
        for si, old_len, new_len, raw_len in changes:
            note = ''
            if new_len == 0:
                note = f'（重编码后 {raw_len} > 桶上限 {B}，已丢弃）'
            print(f'  sample {si}: {old_len} -> {new_len}{note}')
        tmp_tok = _atomic_write_tokens(toks_path, out)
        tmp_off = _atomic_write_offsets(offs_path, new_offs)
        # 旧文件先改名备份，再替换；任何一步崩溃都可 --restore
        os.replace(toks_path, toks_path + '.orig')
        os.replace(offs_path, offs_path + '.orig')
        os.replace(tmp_tok, toks_path)
        os.replace(tmp_off, offs_path)
        manifest['buckets'][bstr]['num_samples'] = int(len(new_offs) - 1)
        manifest['buckets'][bstr]['num_tokens'] = int(len(out))
        _write_manifest(manifest_path, manifest)
        print(f'✓ bucket {B}: tokens {len(toks):,} -> {len(out):,}')

    if not args.apply:
        print(f'\n干跑完成：{total_positions} 个特殊 id，{total_changes} 个样本待修复。'
              f' 加 --apply 执行。')
        return

    print('\n校验中...')
    if _verify(args.flat, manifest, plain):
        moved = _archive_backups(args.flat)
        print('✓ 校验通过：缓存中已无 id 2/3，offsets/大小/桶上限全部一致')
        print(f'  旧文件备份已移到 {os.path.join(args.flat, "repair_backup")}'
              f'（{moved} 个），可用 --restore 回滚；确认无误后可手动删除。')
    else:
        print('❌ 校验失败，请运行 --restore 回滚后检查。')
        sys.exit(2)


if __name__ == '__main__':
    main()
