import numpy as np
from mako.template import Template


def starts_with(words, code):
    for w in words:
        if np.all(np.equal(code, w['enc'][:len(code)])):
            yield w

def shorten_words(words):
    lim=20
    ws = [w['word'] for w in words]
    if len(ws) > lim:
        return ws[:lim]+["..."]
    else:
        return ws

def build_tree_r(words, depth, z_k, code):
    if (not words) or (depth <= 0):
        return None
    node = {'code': code,
            'depth': depth,
            'words': words,
            'shortwords': shorten_words(words),
            'codestr': ''.join(str(c) for c in code),
            'children': []}
    for c in range(z_k):
        tc = code + [c]
        ws = list(starts_with(words, tc))
        n = build_tree_r(words=ws, depth=depth - 1, z_k=z_k, code=tc)
        if n:
            node['children'].append(n)
    return node


def build_tree(enc, vocab, z_k):
    fvocab = ["UNKNOWN"] + vocab
    assert len(fvocab) == enc.shape[0]
    z_depth = enc.shape[1]
    words = [{'enc': enc[i, :], 'word': fvocab[i]} for i in range(len(fvocab))]
    tree = build_tree_r(words, z_depth, z_k, [])
    return tree


def build_js(enc, vocab, tpl_path, z_k):
    tpl = Template(filename=tpl_path)
    tree = build_tree(enc=enc, vocab=vocab, z_k=z_k)
    text = tpl.render(tree=tree)
    return text
