# -*- coding: utf-8 -*-
"""demo_play.py —— 命令行演示：你 vs Links AI
==================================================
既是可跑的演示，也是接入参考：展示了「求走子 → 应用走子 → 终局判定」
的完整闭环，照抄这三步就能接进任何 UI。

用法:
    python demo_play.py            # 均衡难度，你执白
    python demo_play.py 强力 黑    # 强力难度，你执黑
"""
import sys

import chess

from engine_api import LinksAI, game_over


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "均衡"
    human_white = (sys.argv[2] if len(sys.argv) > 2 else "白") != "黑"

    ai = LinksAI(level=level)              # ① 创建 AI（自动加载 best.pt）
    board = chess.Board()
    print("=== Links AI 演示 ===  难度=%s  你执%s" % (level, "白" if human_white else "黑"))

    while True:
        print()
        print(board.unicode(invert_color=not human_white, borders=True))
        over = game_over(board)            # ③ 终局判定
        if over["over"]:
            print("终局: %s (%s)" % (over["result"], over["reason"]))
            break
        human_turn = (board.turn == chess.WHITE) == human_white
        if human_turn:
            try:
                board.push_uci(input("你的走法 (UCI 如 e2e4): ").strip())
            except Exception:
                print("!! 非法走法，重新输入")
        else:
            print("AI 思考中 ...")
            info = ai.think(board)         # ② 求走子（同步；UI 里改用 request_move）
            print("AI: %s   评估 %s   深度%d  %d节点 %dms"
                  % (info["best_move"], info["score_text"],
                     info["depth"], info["nodes"], info["time_ms"]))
            board.push_uci(info["best_move"])


if __name__ == "__main__":
    main()
