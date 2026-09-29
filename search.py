# -*- coding: utf-8 -*-
"""
search.py —— 轻量 αβ/负极大搜索（纯 CPU 批量前向优化）
=====================================================
思路：CNN 策略头负责挑选根节点候选走法（top-K），
     更深层回应按打分器排序（不耗神经网络前向），
     所有叶子局面收集起来一次批量前向评估价值头，
     再自底向上 negamax 回传：value(node) = max( -value(child) )。

     因此即使 2 核 CPU 也能在 0.1~0.3 秒完成 2~3 层搜索。
     value 约定：value(node) = 当前行棋方视角的胜率估计。
"""
import numpy as np
import chess

import encode as enc
import scorer
from model import normalize_input

MATE_VAL = 1e3          # 将死的传播分值（远大于价值头范围 [-1,1]）
MATERIAL_BLEND = 0.75   # 叶子评估中子力分的混合比例（0~1，越大越相信打分器）
MATERIAL_SCALE = 12.0   # 子力分差归一化尺度（后=9 → 满 12 分算接近必胜）


def _batch_value(net, boards):
    """批量评估叶子局面价值（当前行棋方视角）。
    混合策略: v = (1-a)*CNN价值头 + a*子力分差(打分器)，
    既保留 CNN 学到的局面感，又用可靠子力分托底防止送子。"""
    if not boards:
        return np.zeros(0, dtype=np.float32)
    x = np.stack([normalize_input(enc.encode_board(b))[None] for b in boards])
    _, v = net.predict(x)
    v = v.astype(np.float64)
    a = MATERIAL_BLEND
    for i, b in enumerate(boards):
        mat = scorer.material_balance(b)              # 白方视角
        persp = mat if b.turn == chess.WHITE else -mat
        mat_term = max(-1.0, min(1.0, persp / MATERIAL_SCALE))
        v[i] = (1.0 - a) * v[i] + a * mat_term
    return np.asarray(v, dtype=np.float32)


def policy_probs(net, board: chess.Board, temperature: float = 1.0):
    """返回 (合法走法列表, 对应概率, 价值)。概率已按掩码归一化。"""
    mask = enc.legal_move_mask(board)
    x = normalize_input(enc.encode_board(board))[None, None]
    logits, val = net.predict(x)
    logits = logits[0] - (1.0 - mask) * 1e9          # 非法走法屏蔽
    if temperature <= 0:
        probs = np.zeros_like(logits)
        probs[int(np.argmax(logits))] = 1.0
    else:
        e = np.exp((logits - logits.max()) / temperature) * mask
        probs = e / max(e.sum(), 1e-9)
    moves = list(board.legal_moves)
    idx = [enc.move_to_index(m, board) for m in moves]
    return moves, np.array([probs[i] for i in idx]), float(val[0])


def _ordered_replies(board: chess.Board, limit: int):
    """用打分器（不走神经网络）给回应排序，取前 limit 个，省前向开销。"""
    scored = sorted(board.legal_moves,
                    key=lambda m: scorer.score_move(board, m), reverse=True)
    return scored[:limit]


def _terminal_value(b: chess.Board):
    """终局价值（当前行棋方视角）。非终局返回 None。"""
    if b.is_checkmate():
        return -MATE_VAL                    # 当前行棋方被将死
    if b.is_game_over() or b.can_claim_draw():
        return 0.0                          # 逼和/自然和棋
    return None


def select_move(board: chess.Board, net, depth: int = 2,
                top_k: int = 16, leaf_k: int = 6, temperature: float = 0.0):
    """
    主入口：返回 (最佳走法, 我方视角评估分, 信息dict)。
    depth=1 策略+价值一步评估；depth=2/3 批量展开搜索。
    """
    moves, probs, root_val = policy_probs(net, board, temperature=1.0)
    if not moves:
        return None, 0.0, {}
    order = np.argsort(-probs)
    candidates = [moves[i] for i in order[:top_k]]

    # ---------- 构建搜索树（逐层展开） ----------
    # 根候选层：走完候选后轮到对方，节点价值 = 对方视角
    level0 = []
    for m in candidates:
        board.push(m)
        level0.append(dict(path=[m], board=board.copy()))
        board.pop()

    cur = level0
    for d in range(1, max(depth, 1)):
        nxt = []
        for node in cur:
            tv = _terminal_value(node["board"])
            if tv is not None:
                node["term"] = tv
                continue
            replies = _ordered_replies(node["board"], leaf_k)
            node["children"] = []
            for r in replies:
                node["board"].push(r)
                child = dict(path=node["path"] + [r], board=node["board"].copy())
                node["board"].pop()
                nxt.append(child)
                node["children"].append(child)
            if not node["children"]:
                node["term"] = 0.0          # 没有回应（理论上已被终局覆盖）
        cur = nxt
        if not cur:
            break

    # ---------- 叶子批量评估 ----------
    leaves = [nd for nd in cur if "term" not in nd and "val" not in nd]
    if leaves:
        vals = _batch_value(net, [nd["board"] for nd in leaves])
        for nd, v in zip(leaves, vals):
            nd["val"] = float(v)

    # ---------- 自底向上 negamax ----------
    def eval_node(node):
        if "term" in node:
            return node["term"]
        if "val" in node:
            return node["val"]
        if node.get("children"):
            return max(-eval_node(c) for c in node["children"])
        return 0.0

    cand_scores = [-eval_node(nd) for nd in level0]   # 对方视角 -> 我方视角
    best_i = int(np.argmax(cand_scores))
    top = sorted(zip(candidates, cand_scores), key=lambda t: -t[1])[:5]
    info = {"root_value": root_val,
            "best_score": float(cand_scores[best_i]),
            "candidates": [(m.uci(), float(s)) for m, s in top]}
    return candidates[best_i], float(cand_scores[best_i]), info


def sample_move(board: chess.Board, net, temperature: float = 1.0):
    """自我对弈用：按策略概率采样一步（含探索）。返回 (move, 索引, 价值)。"""
    moves, probs, val = policy_probs(net, board, temperature=temperature)
    if not moves:
        return None, -1, val
    i = int(np.random.choice(len(moves), p=probs))
    return moves[i], enc.move_to_index(moves[i], board), val
