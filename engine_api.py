# -*- coding: utf-8 -*-
"""
engine_api.py —— Links AI 对外统一接入接口（唯一需要 import 的文件）
====================================================================
给任何 UI（Kivy / PyQt / Web / 游戏引擎 / 控制台）接Links 引擎用。

三种典型用法：

    # ① 最快上手（同步）
    from engine_api import LinksAI
    ai = LinksAI()                       # 自动加载 models/best.pt
    uci = ai.move("start_fen_or_Board")  # 返回 "e2e4" 这样的 UCI 走法

    # ② UI 线程异步（不卡界面）
    ai = LinksAI(level="强力")
    token = ai.request_move(board, callback)   # callback(uci, score_cp, info)
    ai.cancel()                                # 悔棋/新局时作废旧请求

    # ③ 一次性便捷函数
    from engine_api import quick_move
    uci = quick_move(fen, level="均衡")

走法一律使用 UCI 协议字符串（如 "e2e4"、"e7e8q"），
输入接受 chess.Board 或 FEN 字符串，二者都行。
"""
import os
import sys
import threading

import chess

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)          # 保证 engine.py/search2.py 可被导入

import engine as _drv                  # 隐形驱动：后台线程 + 难度档
from search2 import AlphaBetaEngine    # 主搜索引擎

DEFAULT_MODEL = os.path.join(_HERE, "models", "best.pt")

# 难度 -> (最大深度, 思考毫秒)，与 engine.Player 一致
LEVELS = {"简单": (2, 200), "均衡": (4, 700), "强力": (6, 2200)}


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model(path: str = DEFAULT_MODEL):
    """加载 CNN 权重为推理模式。返回 ChessNet；失败抛异常。"""
    import torch
    import model as _M
    net = _M.build_model()
    sd = torch.load(path, map_location="cpu")
    net.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    net.eval()
    return net


def _to_board(x) -> chess.Board:
    """chess.Board / FEN 字符串 → chess.Board。"""
    if isinstance(x, chess.Board):
        return x
    if isinstance(x, str):
        return chess.Board(x)          # 直接接受 FEN
    raise TypeError("需要 chess.Board 或 FEN 字符串，得到 %r" % type(x))


def to_white_pov(score_cp: int, board: chess.Board) -> int:
    """把引擎的『当前行棋方视角』分数转成白方视角（画评估条用）。"""
    return score_cp if board.turn == chess.WHITE else -score_cp


def score_text(score_cp: int) -> str:
    """百分兵分 → 可读文本（+1.35 / M5）。正值 = 行棋方占优。"""
    from search2 import format_score
    return format_score(score_cp)


def legal_moves(x):
    """所有合法走法（UCI 字符串列表）。"""
    return [m.uci() for m in _to_board(x).legal_moves]


def game_over(x) -> dict:
    """终局判定：给 UI 直接用。
    返回 {"over": bool, "result": "1-0"/"0-1"/"1/2-1/2"/None, "reason": str}"""
    b = _to_board(x)
    if b.is_checkmate():
        r = "0-1" if b.turn == chess.WHITE else "1-0"
        return {"over": True, "result": r, "reason": "checkmate"}
    if b.is_stalemate():
        return {"over": True, "result": "1/2-1/2", "reason": "stalemate"}
    if b.is_insufficient_material():
        return {"over": True, "result": "1/2-1/2", "reason": "insufficient_material"}
    if b.can_claim_fifty_moves():
        return {"over": True, "result": "1/2-1/2", "reason": "fifty_moves"}
    if b.can_claim_threefold_repetition():
        return {"over": True, "result": "1/2-1/2", "reason": "threefold_repetition"}
    if b.is_game_over():
        return {"over": True, "result": "1/2-1/2", "reason": "other"}
    return {"over": False, "result": None, "reason": ""}


# ---------------------------------------------------------------------------
# 主接口类
# ---------------------------------------------------------------------------
class LinksAI:
    """Links 国际象棋 AI 的一行式封装。

    参数:
        model_path: .pt 权重路径，默认包内 models/best.pt
        level:      "简单" / "均衡" / "强力"（默认 均衡）
        think_ms:   覆盖该难度的思考毫秒数（None=用难度默认）
        net:        已加载的 ChessNet（多个 AI 实例共享权重时传入，省内存）
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, level: str = "均衡",
                 think_ms: int | None = None, net=None):
        self.net = net if net is not None else load_model(model_path)
        self.level = level
        self.think_ms = think_ms
        self._lock = threading.Lock()

    # ---------------- 参数 ----------------
    def set_level(self, level: str = "均衡", think_ms: int | None = None):
        """切换难度档 / 思考时间。"""
        self.level = level
        self.think_ms = think_ms

    def _depth_time(self, depth, time_ms):
        d, t = LEVELS.get(self.level, LEVELS["均衡"])
        if self.think_ms is not None:
            t = self.think_ms
        if time_ms is not None:
            t = time_ms
        if depth is not None:
            d = depth
        return d, t

    # ---------------- 同步接口 ----------------
    def move(self, board, depth: int | None = None,
             time_ms: int | None = None) -> str | None:
        """同步求走子（阻塞当前线程直到算完）。
        返回 UCI 字符串（如 "e2e4"）；已终局/无合法走法返回 None。"""
        b = _to_board(board).copy(stack=False)
        d, t = self._depth_time(depth, time_ms)
        with self._lock:
            info = AlphaBetaEngine(net=self.net).search(b, depth=d, time_ms=t)
        mv = info["best_move"]
        return None if mv is None else mv.uci()

    def think(self, board, depth: int | None = None,
              time_ms: int | None = None) -> dict:
        """同 move()，但返回完整分析信息：
            {best_move: "e2e4", score_cp: 35, score_text: "+0.35",
             depth: 4, nodes: 21000, time_ms: 520, pv: ["e2e4", ...]}
        score_cp 为『当前行棋方』视角的百分兵分（越大越好）。"""
        b = _to_board(board).copy(stack=False)
        d, t = self._depth_time(depth, time_ms)
        with self._lock:
            info = AlphaBetaEngine(net=self.net).search(b, depth=d, time_ms=t)
        mv = info["best_move"]
        return {
            "best_move": None if mv is None else mv.uci(),
            "score_cp": info["score_cp"],
            "score_text": score_text(info["score_cp"]),
            "depth": info["depth"], "nodes": info["nodes"],
            "time_ms": info["time_ms"], "pv": info["pv"],
        }

    def evaluate(self, board, depth: int = 2, time_ms: int = 300) -> dict:
        """快速局面评估（不做完整搜索）：
            {"score_cp": 当前行棋方视角, "score_cp_white": 白方视角,
             "best_move": 引擎认为的最佳走法 UCI}
        想要更准可把 depth/time_ms 调大。"""
        info = self.think(board, depth=depth, time_ms=time_ms)
        b = _to_board(board)
        return {"score_cp": info["score_cp"],
                "score_cp_white": to_white_pov(info["score_cp"], b),
                "best_move": info["best_move"]}

    # ---------------- 异步接口（UI 线程专用） ----------------
    def request_move(self, board, callback) -> int:
        """后台线程思考，绝不卡 UI 线程（底层复用 engine.Player）。
        callback(uci, score_cp, info) 会在子线程被调用；
          uci 为 UCI 字符串（失败为 None），info 为 think() 同款完整 dict。
        返回请求 token；悔棋/新局请先 cancel() 再发起新请求。"""
        b = _to_board(board).copy(stack=False)

        def _cb(mv, sc, info):
            if info and info.get("error"):
                callback(None, 0, {"error": True})
                return
            full = dict(info or {})
            full["best_move"] = None if mv is None else mv.uci()
            full["score_text"] = score_text(sc)
            callback(full["best_move"], sc, full)

        self._player = _drv.Player(net=self.net, level=self.level,
                                   think_ms=self.think_ms)
        return self._player.request_move(b, _cb)

    def cancel(self):
        """作废尚未返回的异步请求（悔棋、开新局、关窗口时调用）。"""
        if getattr(self, "_player", None) is not None:
            self._player.cancel()


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
_ai_singleton = None
_ai_lock = threading.Lock()


def quick_move(fen_or_board, level: str = "均衡",
               model_path: str = DEFAULT_MODEL) -> str | None:
    """最省事的一次性接口：传 FEN/Board，拿 UCI 走法。
    内部用全局单例（模型只加载一次，多局复用）。"""
    global _ai_singleton
    with _ai_lock:
        if _ai_singleton is None:
            _ai_singleton = LinksAI(model_path=model_path, level=level)
        else:
            _ai_singleton.set_level(level)
        return _ai_singleton.move(fen_or_board)
