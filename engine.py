# -*- coding: utf-8 -*-
"""engine.py —— 对弈引擎薄封装：模型加载 + 后台思考线程 + 难度档
UI 线程绝不直接跑搜索；所有走子通过回调返回主线程。"""
import threading
import traceback

import chess

import search2
from search2 import AlphaBetaEngine

# 难度 -> (最大深度, 思考毫秒)
LEVELS = {"简单": (2, 200), "均衡": (4, 700), "强力": (6, 2200)}


class Player:
    """封装一个可后台思考的 AI 棋手。每实例独享引擎（TT/killer 不串线）。"""

    def __init__(self, net=None, level: str = "均衡", think_ms: int | None = None):
        self.net = net
        self.level = level
        self.think_ms = think_ms
        self._token = 0            # 请求令牌：换局/悔棋后作废旧请求
        self._busy = False

    def request_move(self, board: chess.Board, callback):
        """异步求走子：callback(move, score_cp, info) 在新线程被调用。
        返回本次请求 token，便于取消。"""
        self._token += 1
        token = self._token
        snap = board.copy()
        t = threading.Thread(target=self._work, args=(snap, token, callback),
                             daemon=True)
        t.start()
        return token

    def cancel(self):
        self._token += 1            # 旧线程照常算完但结果被丢弃

    def _work(self, board: chess.Board, token: int, callback):
        try:
            depth, tms = LEVELS.get(self.level, LEVELS["均衡"])
            if self.think_ms is not None:
                tms = self.think_ms
            eng = AlphaBetaEngine(net=self.net)
            info = eng.search(board, depth=depth, time_ms=tms)
            if token != self._token:
                return              # 已过期（换局/悔棋），丢弃
            callback(info["best_move"], info["score_cp"], info)
        except Exception:
            traceback.print_exc()
            try:
                callback(None, 0, {"error": True})
            except Exception:
                pass


def pick_reference_move(board: chess.Board, kind: str):
    """验证赛用的参考棋手（一次性同步调用）。"""
    import scorer, random
    if kind == "random":
        return random.choice(list(board.legal_moves))
    if kind == "greedy":
        return scorer.greedy_move(board)
    raise ValueError(kind)
