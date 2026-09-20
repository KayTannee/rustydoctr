"""Word polygon scoring with maximum-cardinality one-to-one IoU matching."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from rapidfuzz.distance import Levenshtein


def polygon(box):
    b = np.asarray(box).reshape(-1, 2)
    if len(b) == 2:
        (x0, y0), (x1, y1) = b
        b = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return Polygon(b).buffer(0)


def score_words(truth, predicted, threshold=0.5, return_pairs=False):
    gt = [polygon(w['polygon']) for w in truth]
    pr = [polygon(w['polygon']) for w in predicted]
    tree = STRtree(gt)
    rows, cols = [], []
    merge_counts = np.zeros(len(pr), dtype=int)
    split_counts = np.zeros(len(gt), dtype=int)
    for j, p in enumerate(pr):
        if p.area <= 0:
            continue
        for i in tree.query(p):
            g = gt[i]
            intersection = p.intersection(g).area
            union = p.area + g.area - intersection
            if union and intersection / union >= threshold:
                rows.append(int(i)); cols.append(j)
            if g.area and intersection / g.area >= 0.5:
                merge_counts[j] += 1
            if intersection / p.area >= 0.5:
                split_counts[i] += 1
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(gt), len(pr)))
    match = maximum_bipartite_matching(graph, perm_type='column') if gt and pr else np.full(len(gt), -1)
    pairs = [(i, int(j)) for i, j in enumerate(match) if j >= 0]
    exact = sum(truth[i]['text'] == predicted[j].get('text') for i, j in pairs)
    result = {'truth': len(gt), 'predicted': len(pr), 'matched': len(pairs), 'exact': exact,
            'merge_candidates': int(sum(merge_counts > 1)), 'split_candidates': int(sum(split_counts > 1))}
    if return_pairs:
        result['pairs'] = pairs
    return result


def detection_summary(counts):
    c = {k: sum(x[k] for x in counts) for k in ('truth', 'predicted', 'matched', 'exact', 'merge_candidates', 'split_candidates')}
    precision = c['matched'] / max(c['predicted'], 1)
    recall = c['matched'] / max(c['truth'], 1)
    return dict(c, precision=precision, recall=recall, f1=2*precision*recall/max(precision+recall, 1e-12),
                end_to_end_exact_recall=c['exact']/max(c['truth'], 1))


def recognition_score(truth, predictions):
    assert len(truth) == len(predictions)
    edits = sum(Levenshtein.distance(a, b) for a, b in zip(truth, predictions))
    return {'words': len(truth), 'exact_accuracy': sum(a == b for a, b in zip(truth, predictions))/max(len(truth), 1),
            'casefold_accuracy': sum(a.casefold() == b.casefold() for a, b in zip(truth, predictions))/max(len(truth), 1),
            'cer': edits/max(sum(map(len, truth)), 1), 'character_edits': edits}
