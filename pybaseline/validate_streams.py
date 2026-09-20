"""Audit saved pipeline counts, serialized sequences and timing arithmetic."""
import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path


def validate(root):
    count = 0
    for path in sorted(root.glob('*/summary.json')):
        r = json.loads(path.read_text())
        n = r['pages']
        assert n > 0 and n == r['feeder']['pages'] == r['post']['pages'], path
        assert sum(b['pages'] for b in r['batches']) == n, path
        assert r['post']['timeline'][-1]['pages_total'] == n, path
        assert r['feeder']['wall_seconds'] >= r['config']['seconds'], path
        assert r['pipeline_wall_seconds'] >= r['config']['seconds'], path
        assert abs(n / r['pipeline_wall_seconds'] - r['pages_per_second']) < 1e-9, path
        latency = r['post']['latency_seconds']
        assert 0 < latency['p50'] <= latency['p95'] <= latency['p99'], path
        lines = 0
        text_variants = Counter()
        word_counts = Counter()
        with (path.parent / 'pages.jsonl').open(encoding='utf-8') as stream:
            for sequence, line in enumerate(stream):
                item = json.loads(line)
                assert item['sequence'] == sequence, (path, sequence)
                assert isinstance(item['page']['blocks'], list), (path, sequence)
                words = [word['value'] for block in item['page']['blocks']
                         for line in block['lines'] for word in line['words']]
                signature = hashlib.sha256(json.dumps(words, ensure_ascii=False).encode()).hexdigest()
                text_variants[signature] += 1
                word_counts[len(words)] += 1
                lines += 1
        assert lines == n, path
        assert (path.parent / 'pages.jsonl').stat().st_size == r['post']['output_bytes'], path
        audit = {'pages_checked': lines, 'sequence_count_timing_passed': True,
                 'distinct_word_text_sequences': len(text_variants),
                 'text_sequence_frequencies': dict(text_variants),
                 'word_count_frequencies': dict(word_counts),
                 'note': 'Text variation is diagnostic, not an error: partial final batches can differ numerically.'}
        (path.parent / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(f'{path.parent.name}: {n} pages, {r["pipeline_wall_seconds"]:.1f}s, sequence/count/timing checks passed')
        count += 1
    assert count, 'No completed streams'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('pybaseline/results/streaming'))
    validate(parser.parse_args().root)
