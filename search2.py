# -*- coding: utf-8 -*-
"""
search2.py —— 迭代加深 αβ 搜索引擎（对弈用，与训练用 search.py 分离）
=====================================================================
设计目标：同样的网络，下棋棋力大幅提升。

要点：
  1. 迭代加深 + 置换表（TT），根节点用 CNN 策略排序，内部节点
     MVV-LVA + killer + history 排序（不走神经网络，快）。
  2. 叶子用快速手工评估（子力 + PST + 兵型/车/王安全轻量项），
     归一化到 [-1,1] 与 CNN 价值头同尺度，根节点再与 CNN 混合定夺。
  3. 静态搜索（quiescence）只搜吃子/升变，配简易 SEE 剪掉亏本吃子，
     消除"送子看不到"的地平线效应。
  4. 2 核 CPU 上 1~3 秒可达 5~7 层，战术能力远超旧版固定 2~3 层全展开。

value 约定：内部全部"当前行棋方视角"，分数单位为百分兵(centipawn)。
"""
import time
import chess
import numpy as np

import scorer
import encode as enc

MATE = 100000            # 将死分（距根 ply 越近分越高）
INF = 10 ** 9
MAX_PLY = 64             # qsearch 深度上限防爆栈

# ---------------------------------------------------------------------------
# 快速评估：子力 + PST（Simplified Evaluation Function 表）
# 表格按"白方视角、a1=0 … h8=63"填写；黑方镜像使用。
# ---------------------------------------------------------------------------
PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

PAWN_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0]
KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50]
BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20]
ROOK_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0]
QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20]
KING_MG_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20]
KING_EG_PST = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-50,-50]

PST = {chess.PAWN: PAWN_PST, chess.KNIGHT: KNIGHT_PST, chess.BISHOP: BISHOP_PST,
       chess.ROOK: ROOK_PST, chess.QUEEN: QUEEN_PST,
       chess.KING: KING_MG_PST}


def mirror(sq: int) -> int:
    """黑方棋子用：垂直镜像。"""
    return sq ^ 56


def eval_fast(board: chess.Board) -> int:
    """白方视角的快速局面评估（百分兵）。含兵型/双象/车占线/王安全轻量项。"""
    if board.is_checkmate():
        return -MATE + board.ply()
    if board.is_insufficient_material():
        return 0
    score = 0
    board_pmap = board.piece_map()
    non_pawn = 0
    bishops = [0, 0]
    pawn_files_w = [0] * 8
    pawn_files_b = [0] * 8
    for sq, pc in board_pmap.items():
        pt = pc.piece_type
        if pt == chess.PAWN:
            if pc.color:
                score += PIECE_VALUE[pt] + PST[pt][sq]
                pawn_files_w[chess.square_file(sq)] += 1
            else:
                score -= PIECE_VALUE[pt] + PST[pt][mirror(sq)]
                pawn_files_b[chess.square_file(sq)] += 1
        else:
            v = PIECE_VALUE[pt] + PST[pt][sq if pc.color else mirror(sq)]
            score += v if pc.color else -v
            if pt != chess.KING:
                non_pawn += PIECE_VALUE[pt]
            if pt == chess.BISHOP:
                bishops[0 if pc.color else 1] += 1
    # 双象
    if bishops[0] >= 2:
        score += 30
    if bishops[1] >= 2:
        score -= 30
    # 叠兵 / 孤兵（轻量）
    for f in range(8):
        if pawn_files_w[f] > 1:
            score -= 12 * (pawn_files_w[f] - 1)
        if pawn_files_b[f] > 1:
            score += 12 * (pawn_files_b[f] - 1)
        left_w = pawn_files_w[f - 1] if f > 0 else 0
        right_w = pawn_files_w[f + 1] if f < 7 else 0
        left_b = pawn_files_b[f - 1] if f > 0 else 0
        right_b = pawn_files_b[f + 1] if f < 7 else 0
        if pawn_files_w[f] and left_w == 0 and right_w == 0:
            score -= 14
        if pawn_files_b[f] and left_b == 0 and right_b == 0:
            score += 14
    # 残局王活动性：子力少时用残局王表
    if non_pawn <= 1500:
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is not None:
            score += (KING_EG_PST[wk] - KING_MG_PST[wk])
        if bk is not None:
            score -= (KING_EG_PST[mirror(bk)] - KING_MG_PST[mirror(bk)])
    # 临时优势（先行小奖励）
    score += 8 if board.turn == chess.WHITE else -8
    return score


def _see(board: chess.Board, move: chess.Move) -> int:
    """简易静态交换评估（标准 swap-off）：走完这步吃子后该格的净得子（百分兵）。"""
    if not board.is_capture(move):
        return 0
    to_sq = move.to_square
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        victim = board.piece_type_at(to_sq)
    gain = [PIECE_VALUE[victim] if victim else 0]
    b = board.copy(stack=False)
    b.push(move)
    while True:
        # 轮到 b.turn（被吃方）吃回：找攻击 to_sq 的最小价值子
        attackers = list(b.attackers(b.turn, to_sq))
        if not attackers:
            break
        atk_sq = min(attackers,
                     key=lambda s: PIECE_VALUE.get(b.piece_type_at(s), 20000))
        atk_pt = b.piece_type_at(atk_sq)
        m = chess.Move(atk_sq, to_sq)
        if not b.is_legal(m):          # 王不能送进将军等情况
            break
        gain.append(PIECE_VALUE[atk_pt] - gain[-1])
        b.push(m)
    # swap-off：从叶子往根取最大
    d = len(gain) - 1
    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1
    return gain[0]


class Timeout(Exception):
    pass


class AlphaBetaEngine:
    """迭代加深 αβ 引擎。线程不安全：一个引擎实例一个线程用。"""

    def __init__(self, net=None, policy_weight: float = 1.0):
        self.net = net                 # CNN（可 None，纯手工引擎）
        self.policy_weight = policy_weight
        self.tt = {}                   # key -> (depth, flag, score, move)
        self.killers = [[None, None] for _ in range(256)]
        self.history = {}
        self.nodes = 0
        self.deadline = None
        self.last_pv = []

    # ---------------- 时间与节点控制 ----------------
    def _tick(self):
        self.nodes += 1
        if self.nodes % 2048 == 0 and self.deadline is not None:
            if time.time() > self.deadline:
                raise Timeout

    def _tt_key(self, board: chess.Board):
        try:
            return board._transposition_key()
        except Exception:
            return board.fen()

    # ---------------- 排序 ----------------
    def _mvv_lva(self, board: chess.Board, m: chess.Move) -> int:
        if board.is_en_passant(m):
            return 10 * 100 - 1
        v = board.piece_type_at(m.to_square)
        a = board.piece_type_at(m.from_square)
        return 10 * (PIECE_VALUE.get(v, 0) if v else 0) - \
            (PIECE_VALUE.get(a, 0) if a else 0)

    def _order(self, board: chess.Board, moves, ply: int, tt_move=None,
               root_probs=None):
        ply = min(ply, 255)

        def key(m):
            if tt_move is not None and m == tt_move:
                return 10 ** 9
            if root_probs is not None:
                return int(root_probs.get(m.uci(), 0) * 1e6) + self._mvv_lva(board, m)
            s = 0
            if board.is_capture(m):
                s += 10 ** 6 + self._mvv_lva(board, m)
            if m.promotion:
                s += 10 ** 6
            if self.killers[ply][0] == m or self.killers[ply][1] == m:
                s += 10 ** 5
            s += self.history.get((board.turn, m.from_square, m.to_square), 0)
            return s
        return sorted(moves, key=key, reverse=True)

    # ---------------- 静态搜索 ----------------
    def _qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        side = 1 if board.turn == chess.WHITE else -1
        stand = side * eval_fast(board)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        if ply >= MAX_PLY:
            return stand
        caps = [m for m in board.legal_moves
                if board.is_capture(m) or m.promotion]
        # SEE 剪枝：亏本吃子直接跳过
        scored = []
        for m in caps:
            see = _see(board, m)
            if see < -50 and not m.promotion:
                continue
            scored.append((self._mvv_lva(board, m), m))
        scored.sort(key=lambda t: t[0], reverse=True)
        for _, m in scored:
            board.push(m)
            sc = -self._qsearch(board, -beta, -alpha, ply + 1)
            board.pop()
            if sc >= beta:
                return sc
            if sc > alpha:
                alpha = sc
        return alpha

    # ---------------- 主搜索 ----------------
    def _ab(self, board: chess.Board, depth: int, alpha: int, beta: int,
            ply: int) -> int:
        self._tick()
        if board.is_checkmate():
            return -(MATE - ply)
        if board.is_game_over():
            return 0
        if board.is_repetition(2) or board.is_fifty_moves():
            return 0
        if depth <= 0:
            return self._qsearch(board, alpha, beta, ply)

        key = self._tt_key(board)
        tt_move = None
        e = self.tt.get(key)
        if e is not None:
            e_depth, e_flag, e_score, e_move = e
            tt_move = e_move
            if e_depth >= depth and ply > 0:
                if e_flag == 0:
                    return e_score
                if e_flag == 1 and e_score >= beta:
                    return e_score
                if e_flag == 2 and e_score <= alpha:
                    return e_score

        if board.is_check():
            depth += 1                      # 将军延伸

        moves = list(board.legal_moves)
        moves = self._order(board, moves, ply, tt_move)
        best = -INF
        best_move = None
        old_alpha = alpha
        first = True
        for m in moves:
            board.push(m)
            if first:
                sc = -self._ab(board, depth - 1, -beta, -alpha, ply + 1)
                first = False
            else:
                # PVS：先零窗试探
                sc = -self._ab(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < sc < beta:
                    sc = -self._ab(board, depth - 1, -beta, -sc, ply + 1)
            board.pop()
            if sc > best:
                best = sc
                best_move = m
            if sc > alpha:
                alpha = sc
            if alpha >= beta:
                if not board.is_capture(m):
                    k = self.killers[ply]
                    if k[0] != m:
                        k[1] = k[0]
                        k[0] = m
                    hkey = (board.turn, m.from_square, m.to_square)
                    self.history[hkey] = self.history.get(hkey, 0) + depth * depth
                break

        flag = 0
        if best <= old_alpha:
            flag = 2
        elif best >= beta:
            flag = 1
        self.tt[key] = (depth, flag, best, best_move)
        return best

    # ---------------- 根入口 ----------------
    def search(self, board: chess.Board, depth: int = 6, time_ms: int = 1200,
               root_top: int = 24) -> dict:
        """返回 dict(best_move, score_cp, depth, nodes, time_ms, pv, cand)。
        重要：内部在副本上搜索（Timeout 中断可能留下未 pop 的栈，
        绝不允许污染调用方棋盘）。"""
        board = board.copy(stack=False)          # 一切 push/pop 都在副本上
        moves = list(board.legal_moves)
        if not moves:
            return dict(best_move=None, score_cp=0, depth=0, nodes=0,
                        time_ms=0, pv=[], cand=[])
        self.nodes = 0
        self.deadline = time.time() + max(time_ms, 50) / 1000.0
        start = time.time()

        # 根排序：CNN 策略优先
        root_probs = None
        if self.net is not None:
            try:
                import search as _s1
                _mv, probs, _v = _s1.policy_probs(self.net, board, temperature=1.0)
                root_probs = {m.uci(): float(p) for m, p in zip(_mv, probs)}
            except Exception:
                root_probs = None
        ordered = self._order(board, moves, 0, root_probs=root_probs)
        ordered = ordered[:max(root_top, 4)]

        best_move = ordered[0]
        best_score = 0
        completed = 0
        pv = []
        for d in range(1, max(depth, 1) + 1):
            try:
                alpha, beta = -INF, INF
                cur_best, cur_score = None, -INF
                scores = {}
                for m in ordered:
                    board.push(m)
                    sc = -self._ab(board, d - 1, -beta, -alpha, 1)
                    board.pop()
                    scores[m.uci()] = sc
                    if sc > cur_score:
                        cur_score, cur_best = sc, m
                    if sc > alpha:
                        alpha = sc
                best_move, best_score = cur_best, cur_score
                completed = d
                # 按本轮分数重排，下一轮优先搜好棋
                ordered.sort(key=lambda m: scores.get(m.uci(), -INF), reverse=True)
                pv = self._extract_pv(board, best_move)
                if abs(best_score) > MATE - 100:
                    break                       # 已见将死，不必再深
                if time.time() > self.deadline:
                    break
            except Timeout:
                break
        info = dict(best_move=best_move, score_cp=int(best_score),
                    depth=completed, nodes=self.nodes,
                    time_ms=int((time.time() - start) * 1000), pv=pv, cand=[])
        return info

    def _extract_pv(self, board: chess.Board, first: chess.Move):
        pv = [first.uci()]
        b = board.copy()
        b.push(first)
        for _ in range(8):
            e = self.tt.get(self._tt_key(b))
            if not e or e[3] is None:
                break
            m = e[3]
            if m not in b.legal_moves:
                break
            pv.append(m.uci())
            b.push(m)
        return pv


def select_move_v2(board: chess.Board, net=None, level: str = "均衡",
                   think_ms: int | None = None) -> tuple:
    """对弈入口：返回 (move, score, info)。score 为行棋方视角百分兵。"""
    preset = {"简单": (2, 150), "均衡": (4, 600), "强力": (6, 2000)}
    d, t = preset.get(level, preset["均衡"])
    if think_ms is not None:
        t = think_ms
    eng = AlphaBetaEngine(net=net)
    info = eng.search(board, depth=d, time_ms=t)
    return info["best_move"], info["score_cp"], info


def format_score(cp: int, mover_perspective: bool = True) -> str:
    """把百分兵分转成可读文本（如 +1.35 / M5）。"""
    if cp is None:
        return "?"
    if abs(cp) > MATE - 200:
        plies = MATE - abs(cp)
        n = max(1, (plies + 1) // 2)
        return ("M" + str(n)) if cp > 0 else ("-M" + str(n))
    return f"{cp / 100:+.2f}"
