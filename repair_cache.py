#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复被中断/并发写坏的 flat_u16 缓存（不删除原始数据）。

做法：
  1. 对每个桶，用 lengths_u32 记录的长度序列和 token 文件的实际大小，
     保留“能完整放下”的样本前缀；
  2. 只截掉每个桶末尾残缺的几十/几百字节，不删除整桶；
  3. 依据保留下来的 lengths 重新生成 offsets.npy；
  4. 原子写入 manifest.json / progress.json，使缓存可以直接被 train.py 复用。

用法：
  python repair_cache.py            # 自动发现 ./cache/*/flat_u16
  python repair_cache.py <flat_u16 目录>
"""
import json
import os
import glob
import sys
import time

import numpy as np

DATA_DIR = './data'
CACHE_ROOT = './cache'


def find_flat_dir():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    cands = sorted(glob.glob(os.path.join(CACHE_ROOT, '*', 'flat_u16')))
    cands = [c for c in cands if os.path.exists(os.path.join(c, 'progress.json'))]
    if not cands:
        raise SystemExit('没有找到带 progress.json 的 flat_u16 目录')
    if len(cands) > 1:
        print('发现多个缓存目录:')
        for i, c in enumerate(cands):
            print(f'  [{i}] {c}')
        raise SystemExit('请显式传入要修复的目录')
    return cands[0]


def jsonl_fingerprint():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.jsonl')))
    files = [f for f in files if 'wt.jsonl' not in f and 'bucketed_cache' not in f]
    return [[os.path.basename(f), os.path.getsize(f)] for f in files]


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    flat_dir = find_flat_dir()
    progress_path = os.path.join(flat_dir, 'progress.json')
    manifest_path = os.path.join(flat_dir, 'manifest.json')
    len_dir = os.path.join(flat_dir, 'lengths_u32')

    print(f'修复缓存目录: {flat_dir}')
    with open(progress_path) as f:
        state = json.load(f)

    fp = jsonl_fingerprint()
    if state.get('fingerprint') != fp:
        raise SystemExit('progress.json 里的输入指纹与当前 data/*.jsonl 不一致，已中止。')
    # 桶边界从 progress.json 读取，脚本不绑定具体 max_seq_len。
    bucket_boundaries = sorted(int(k) for k in state['buckets'])

    # 备份 progress.json（保留现场）
    bak = progress_path + time.strftime('.bak_%Y%m%d_%H%M%S')
    with open(progress_path) as f:
        text = f.read()
    with open(bak, 'w') as f:
        f.write(text)
    print(f'已备份 progress.json -> {os.path.basename(bak)}')

    buckets_info = {}
    total_samples = 0
    total_tokens = 0
    dropped_samples = 0
    truncated_bytes = 0

    for b in bucket_boundaries:
        tok_path = os.path.join(flat_dir, f'bucket_{b}.tokens.u16')
        len_path = os.path.join(len_dir, f'bucket_{b}.lengths.u32')
        ta = os.path.getsize(tok_path) if os.path.exists(tok_path) else 0
        la = os.path.getsize(len_path) if os.path.exists(len_path) else 0
        L = la // 4

        if L > 0:
            lengths = np.fromfile(len_path, dtype=np.uint32, count=L).astype(np.int64)
            cum = np.cumsum(lengths) * 2  # 每个样本前缀占用的 token 字节数
            # 第一个“前缀字节数 > token 文件大小”的位置，就是能完整保留的样本数
            k = int(np.searchsorted(cum, ta, side='right'))
            k = min(k, L)
        else:
            lengths = np.zeros(0, dtype=np.int64)
            k = 0

        if k > 0:
            keep_tokens = int(lengths[:k].sum())
            need_bytes = keep_tokens * 2
            # 只截掉末尾不完整的样本字节；正常情况只掉几个字节
            if ta > need_bytes:
                truncated_bytes += ta - need_bytes
                with open(tok_path, 'r+b') as f:
                    f.truncate(need_bytes)
            elif ta < need_bytes:
                k = 0
                keep_tokens = 0
            if k > 0:
                new_la = k * 4
                if la > new_la:
                    with open(len_path, 'r+b') as f:
                        f.truncate(new_la)
                offsets = np.zeros(k + 1, dtype=np.int64)
                np.cumsum(lengths[:k], out=offsets[1:])
                np.save(os.path.join(flat_dir, f'bucket_{b}.offsets.npy'), offsets)
        else:
            keep_tokens = 0

        dropped = L - k
        dropped_samples += dropped
        total_samples += k
        total_tokens += keep_tokens
        buckets_info[str(b)] = {'num_samples': k, 'num_tokens': keep_tokens}
        state['buckets'][str(b)] = {
            'num_samples': k,
            'num_tokens': keep_tokens,
            'token_bytes': keep_tokens * 2 if k > 0 else ta,
            'lengths_bytes': k * 4 if k > 0 else la,
        }
        if dropped:
            print(f'  bucket {b:>4}: 保留 {k:>7,} 样本, 丢弃末尾 {dropped:>6,} 条'
                  f'{" (token 文件已损坏)" if ta < 2 else ""}')

    manifest = {
        'version': 1,
        'source': 'jsonl',
        'dtype': 'uint16',
        'bucket_boundaries': bucket_boundaries,
        'fingerprint': fp,
        'jsonl_fingerprint': fp,
        'buckets': buckets_info,
        'repair': {
            'repaired_at': time.time(),
            'dropped_samples': dropped_samples,
            'truncated_bytes': truncated_bytes,
        },
        'created_at': time.time(),
    }
    atomic_json(manifest_path, manifest)

    state['file_idx'] = len(fp)
    state['file_pos'] = 0
    state['total'] = total_samples
    state['done'] = True
    atomic_json(progress_path, state)

    print('\n修复完成:')
    print(f'  保留样本 : {total_samples:,}')
    print(f'  保留 token: {total_tokens/1e9:.3f}B')
    print(f'  丢弃样本 : {dropped_samples:,} ({dropped_samples/(total_samples+dropped_samples)*100:.3f}%)')
    print(f'  截掉尾部字节: {truncated_bytes:,}')
    print(f'  manifest : {manifest_path}')
    print('\n现在可以直接 python train.py，会跳过重新编码。')


if __name__ == '__main__':
    main()
