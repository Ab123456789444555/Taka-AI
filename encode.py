# -*- coding: utf-8 -*-
"""
encode.py —— 棋盘输入编码器
=====================================
把 python-chess 的棋盘对象转换成 CNN 需要的「带符号整数矩阵」。

编码规则（与用户描述一致，扩展到 8x8）：
    空 = 0
    白方: 兵=1  马=2  象=3  车=4  后=5  王=6
    黑方: 对应取负  -1 ~ -6

视角约定（对 AI 学习至关重要）：
    矩阵永远以「当前行棋方」的视角生成——
    即己方棋子永远是正数、己方底线永远在第 0 行。
    这样 AI 不需要区分自己是白是黑，学起来更快。

例如用户举的例子：3x3 棋盘中间有一个白兵 ->
    000
    010
    000
真实棋盘是 8x8，起始局面是：
    -4 -2 -3 -5 -6 -3 -2 -4
    -1 -1 -1 -1 -1 -1 -1 -1
     0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0
     1  1  1  1  1  1  1  1
     4  2  3  5  6  3  2  4
"""
import numpy as np
import chess

# 棋子 -> 数字 编码表（兵=1 ... 王=6，黑方取负）
PIECE_CODE = {
    chess.PAWN: 1,    # 兵
    chess.KNIGHT: 2,  # 马
    chess.BISHOP: 3,  # 象
    chess.ROOK: 4,    # 车
    chess.QUEEN: 5,   # 后
    chess.KING: 6,    # 王
}
CODE_PIECE = {v: k for k, v in PIECE_CODE.items()}

# 策略头索引空间：起点64格 x 终点64格 = 4096 个输出
POLICY_SIZE = 64 * 64


def _persp_square(sq: int, turn: chess.Color) -> int:
    """把真实格子坐标转换成「当前行棋方视角」的坐标（黑方时上下翻转）。"""
    if turn == chess.BLACK:
        return sq ^ 56  # 翻转行：a8<->a1
    return sq


def encode_board(board: chess.Board) -> np.ndarray:
    """棋盘 -> (8, 8) float32 矩阵（当前行棋方视角，己方正数）。"""
    mat = np.zeros((8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        v = float(PIECE_CODE[piece.piece_type])
        if piece.color == chess.BLACK:
            v = -v
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if board.turn == chess.BLACK:
            rank = 7 - rank
            v = -v
        mat[rank, file] = v
    return mat


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """走法 -> 策略头索引（0~4095）。索引 = 视角起点格 * 64 + 视角终点格。"""
    frm = _persp_square(move.from_square, board.turn)
    to = _persp_square(move.to_square, board.turn)
    return frm * 64 + to


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """策略头索引 -> 真实走法（升变自动按后处理，若该走法合法）。"""
    frm = index // 64
    to = index % 64
    frm = _persp_square(frm, board.turn)
    to = _persp_square(to, board.turn)
    # 找到与 (from, to) 匹配的合法走法（升变时优先选择升后）
    best = None
    for m in board.legal_moves:
        if m.from_square == frm and m.to_square == to:
            if m.promotion in (None, chess.QUEEN):
                return m
            best = best or m
    return best  # 可能是 None（不合法），由调用方处理


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """生成 4096 维合法走法掩码（升变的4种走法合并到同一个索引）。"""
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    for m in board.legal_moves:
        mask[move_to_index(m, board)] = 1.0
    return mask


def matrix_str(board: chess.Board) -> str:
    """把编码矩阵打印成用户示例那种 000/010/000 的文本形式。"""
    mat = encode_board(board)
    lines = []
    for row in mat:
        lines.append(" ".join(f"{int(v):2d}" for v in row))
    return "\n".join(lines)


def encode_batch(boards) -> np.ndarray:
    """批量编码 -> (B, 1, 8, 8)。"""
    return np.stack([encode_board(b)[None, ...] for b in boards])
