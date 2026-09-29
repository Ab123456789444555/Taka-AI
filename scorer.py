# -*- coding: utf-8 -*-
"""
scorer.py —— 走法打分器（规则/价值观引擎）
==========================================
按照用户要求给每一步走法打分，例如：
    · 走中间的兵               ->  +1 分
    · 吃掉对方棋子             ->  + 对方棋子价值
    · 把自己的后送出去被吃掉    ->  - 后的价值（用户示例为 -5，本默认表后=9，
                                   想改成 -5 只需把下面 QUEEN 的分值改成 5 再重新训练）
    · 送车/送马/送兵           ->  - 对应价值（没人保护的棋子会被吃）
    · 王车易位 / 将军 / 升变    ->  各有额外加分
    · 将死对方                 ->  +200 分

这个打分器有 3 个用途：
    1. 训练时给自我对弈的每一步生成「打分奖励」；
    2. 生成策略学习的软标签（AI 模仿高分走法）；
    3. 客户端在没加载模型时作为兜底 AI。
"""
import chess

# ------- 可随意修改的分值表（改完重新训练即可让 AI 适应新的价值观）-------
PIECE_SCORE = {
    chess.PAWN: 1,     # 兵 1 分
    chess.KNIGHT: 3,   # 马 3 分
    chess.BISHOP: 3,   # 象 3 分
    chess.ROOK: 5,     # 车 5 分
    chess.QUEEN: 9,    # 后 9 分（用户示例口径为 5 分，可自行修改）
    chess.KING: 0,     # 王无价（被将死单独计分）
}
CENTER_PAWN_BONUS = 1.0     # 走中间的兵 +1 分（用户指定）
CASTLE_BONUS = 1.0          # 王车易位 +1 分（保护王）
CHECK_BONUS = 0.5           # 将军 +0.5 分
PROMOTION_BONUS = 8.0       # 兵升变（额外奖励，吃子分另算）
DEVELOP_BONUS = 0.3         # 出马/出象（离开初始位置）+0.3 分
MATE_SCORE = 200.0          # 将死 +200 分
HANG_SCALE = 1.0            # 送吃惩罚系数（1.0 = 全额扣子力价值）
# 中央四格 d4 e4 d5 e5（0-index 文件/行号 3、4）
_CENTER_FILES = (3, 4)
_CENTER_RANKS = (3, 4)


def _value(board: chess.Board, sq: int) -> float:
    p = board.piece_type_at(sq)
    return PIECE_SCORE.get(p, 0) if p else 0.0


KING_ATTACK_VAL = 100.0     # 王作为攻击者时的虚拟分值（王不会白换子）


def _hanging_penalty(board_after: chess.Board, move: chess.Move) -> float:
    """
    简化版交换评估（SEE-lite）：走完这步后，走到的格子是否会被对方吃掉？
    注意：调用时已经 push，board_after.turn = 回应方（对方）。
    返回正数表示惩罚值（调用方扣分）。
        · 目标格无保护           -> 白送，扣全部子力价值（送后 -> -9）
        · 有保护但对方攻击子更小  -> 换子亏，扣差价的一半
    """
    them = board_after.turn          # 对方（即将行棋的一方）
    us = not them                    # 己方（刚走完棋的一方）
    to = move.to_square
    moved = board_after.piece_type_at(to)
    if moved is None:
        return 0.0
    moved_val = PIECE_SCORE.get(moved, 0)
    if not board_after.is_attacked_by(them, to):
        return 0.0                   # 对方吃不到，安全
    attackers = board_after.attackers(them, to)
    if not attackers:
        return 0.0
    # 王参与攻击时按虚拟高分计（有保护时王不会吃）
    vals = []
    for a in attackers:
        pt = board_after.piece_type_at(a)
        vals.append(KING_ATTACK_VAL if pt == chess.KING else PIECE_SCORE.get(pt, 0))
    min_att = min(vals)
    defended = board_after.is_attacked_by(us, to)   # 己方是否有保护
    if not defended:
        return HANG_SCALE * moved_val               # 直接白丢（送后 -> -9）
    if min_att >= KING_ATTACK_VAL:
        return 0.0                                  # 只有王能吃且己方有保护 -> 王不会吃
    if min_att < moved_val:
        return HANG_SCALE * (moved_val - min_att) * 0.5  # 换子亏差价
    return 0.0


def score_move(board: chess.Board, move: chess.Move) -> float:
    """给「当前局面下的一步合法走法」打分（从行棋方视角，正分好、负分差）。"""
    s = 0.0
    captured = board.piece_type_at(move.to_square)
    if board.is_en_passant(move):
        captured = chess.PAWN
    if captured:
        s += PIECE_SCORE[captured]            # 吃子加分（吃后 +9）

    if board.is_castling(move):
        s += CASTLE_BONUS                     # 易位保护王

    moved = board.piece_type_at(move.from_square)
    if moved == chess.PAWN:
        if move.promotion:
            s += PROMOTION_BONUS              # 升变 +8
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if to_file in _CENTER_FILES and to_rank in _CENTER_RANKS:
            s += CENTER_PAWN_BONUS            # 走中间的兵 +1（用户指定）
    elif moved in (chess.KNIGHT, chess.BISHOP):
        home_rank = 0 if board.turn == chess.WHITE else 7
        if chess.square_rank(move.from_square) == home_rank:
            s += DEVELOP_BONUS                # 轻子出动

    if board.gives_check(move):
        s += CHECK_BONUS                      # 将军

    # 推演一步看后果：将死 / 送吃
    board.push(move)
    try:
        if board.is_checkmate():
            s += MATE_SCORE                   # 直接将死 +200
        else:
            s -= _hanging_penalty(board, move)
    finally:
        board.pop()
    return s


def score_all_moves(board: chess.Board) -> dict:
    """给所有合法走法打分。返回 {策略头索引: 分数}。"""
    from encode import move_to_index
    return {move_to_index(m, board): score_move(board, m) for m in board.legal_moves}


def greedy_move(board: chess.Board):
    """打分器贪心走法（评估基线 AI / 未加载模型时的兜底）。"""
    best, best_s = None, -1e9
    for m in board.legal_moves:
        s = score_move(board, m)
        if s > best_s:
            best, best_s = m, s
    return best, best_s


def material_balance(board: chess.Board) -> float:
    """白方视角的场面子力分差（用于无结果对局的判定与评估条）。"""
    total = 0.0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        v = PIECE_SCORE[p.piece_type]
        total += v if p.color == chess.WHITE else -v
    return total
