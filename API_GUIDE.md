# Links AI 核心引擎 · 接入文档（API_GUIDE）

> 版本 1.0 ｜ 适用包：`Links_Core`
> 一句话：**读完全文约 10 分钟，照抄示例即可把 Links AI 接进你自己的 UI。**

---

## 1. 这个包是什么

`Links_Core` 是 Links 国际象棋的**纯引擎包**：已删除全部界面代码（无 Kivy、无窗口、无字体资源），
只保留「搜索引擎 + AI 模型 + 无界面驱动 + 统一接入层」。你的 UI 负责画棋盘和收鼠标，
走子决策全部交给本包。

```
你的 UI  ──>  engine_api.LinksAI   （你唯一需要 import 的文件）
                 │
                 ├─> engine.Player      无界面驱动：后台线程 + 难度档 + 取消机制
                 ├─> search2.AlphaBetaEngine   主搜索：迭代加深 αβ + 置换表
                 │      + killer/history + 静态搜索(qsearch) + SEE 剪枝
                 │      根节点用 CNN 策略排序
                 ├─> search.py          轻量引擎 + policy_probs（search2 的依赖）
                 └─> model.py + encode.py + scorer.py + models/best.pt
                        CNN 双头网络：策略头(4096 走法) + 价值头([-1,1])
```

## 2. 包内文件清单

| 文件 | 作用 | 你需要动吗 |
|---|---|---|
| `engine_api.py` | **统一接入接口**（LinksAI 类 + 便捷函数） | 只 import，不改 |
| `engine.py` | 无界面驱动：`Player` 后台思考线程、难度档 | 只 import，不改 |
| `search2.py` | 主搜索引擎（对弈强度来源） | 不改 |
| `search.py` | 轻量搜索引擎；提供 `policy_probs` 供 search2 根排序 | 不改 |
| `model.py` | ChessNet 网络定义（约 140 万参数） | 不改 |
| `encode.py` | 棋盘 → 神经网络输入编码 | 不改 |
| `scorer.py` | 规则打分器 / 子力分（引擎底层依赖） | 不改 |
| `models/best.pt` | **AI 模型权重**（对抗赛胜率最高的 checkpoint） | 不改 |
| `demo_play.py` | 命令行演示（也是接入参考代码） | 可跑/可抄 |
| `requirements.txt` | 依赖清单（无任何 UI 依赖） | — |

## 3. 环境要求与安装

- Python ≥ 3.10
- 三个依赖，无 UI 框架要求：

```bash
pip install python-chess numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版即可
```

## 4. 30 秒上手

```python
from engine_api import LinksAI

ai = LinksAI()                       # 默认: models/best.pt, 难度"均衡"
uci = ai.move(chess.Board())         # 传 FEN 字符串也行: ai.move("rnbq... w - - 0 1")
print(uci)                           # "e2e4" —— UCI 走法字符串，直接应用到你的棋盘
```

验证安装（可选）：

```bash
python demo_play.py        # 命令行跟 AI 下一盘
python demo_play.py 强力 黑 # 强力难度、你执黑
```

---

## 5. 核心 API

### 5.1 `LinksAI(...)` —— 创建 AI 实例

```python
ai = LinksAI(model_path="models/best.pt", level="均衡", think_ms=None, net=None)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_path` | 包内 `models/best.pt` | 权重路径。多实例共享权重时见 5.7 |
| `level` | `"均衡"` | `"简单"` / `"均衡"` / `"强力"` |
| `think_ms` | `None` | 覆盖该难度的思考毫秒数；`None` 用难度默认 |
| `net` | `None` | 传入已加载的 ChessNet（`load_model()` 的返回值），多实例省内存 |

**难度档表：**

| level | 最大深度 | 思考时间 | 定位 |
|---|---|---|---|
| `简单` | 2 层 | 200ms | 给新手陪练 |
| `均衡` | 4 层 | 700ms | 默认，约 0.5~0.7 秒/步 |
| `强力` | 6 层 | 2200ms | 最强，约 2~3 秒/步 |

### 5.2 `ai.move(board, depth=None, time_ms=None) -> str|None` —— 同步求走子

```python
uci = ai.move(board)            # 阻塞直到算完，返回 "e2e4" / "e7e8q" 等
uci = ai.move(fen, time_ms=200) # 临时快棋：200ms 一手
```

- 输入：`chess.Board` 或 FEN 字符串，均可。
- 返回：UCI 字符串；**已终局 / 无合法走法时返回 `None`**（先用 `game_over()` 判定更稳）。
- `depth` / `time_ms` 可单步临时覆盖难度档，不影响 `ai` 之后的默认设置。

### 5.3 `ai.think(board, depth=None, time_ms=None) -> dict` —— 走法 + 完整分析

```python
info = ai.think(board)
# {"best_move": "e2e4", "score_cp": 35, "score_text": "+0.35",
#  "depth": 4, "nodes": 21384, "time_ms": 612, "pv": ["e2e4","e7e5","g1f3",...]}
```

| 字段 | 含义 |
|---|---|
| `best_move` | 引擎选择的走法（UCI 字符串；无合法走法为 `None`） |
| `score_cp` | **当前行棋方视角**的百分兵分（+35 = 行棋方好约 0.35 兵） |
| `score_text` | 可读文本：`+1.35` / `-0.40`；将死时为 `M5`（5 步内杀）/ `-M3`（被 3 步杀） |
| `depth` | 实际完成的搜索深度 |
| `nodes` | 搜索节点数 |
| `time_ms` | 实际耗时 |
| `pv` | 主变（principal variation），走法 UCI 列表，可用于"AI在想什么"展示 |

> 评估条画法：要把分数转成**白方视角**（条形图惯例），用 `to_white_pov(score_cp, board)`。

### 5.4 `ai.evaluate(board, depth=2, time_ms=300) -> dict` —— 快速局面评估

```python
ev = ai.evaluate(board)
# {"score_cp": -12, "score_cp_white": 12, "best_move": "g1f3"}
```

只做浅层搜索换取速度，适合局面预览、复盘标注。想更准就调大 `depth/time_ms`。

### 5.5 `ai.request_move(board, callback) -> token` ／ `ai.cancel()` —— 异步（UI 线程专用）

**UI 里绝不要在主线程调 `move()/think()`**（会卡住界面几百毫秒到几秒）。
用异步接口，引擎在后台线程思考，结果通过回调回到你这里：

```python
def on_ai_move(uci, score_cp, info):
    # 在子线程被调用！刷新界面请按你的框架规则切回主线程
    apply_move(uci)                     # 你的棋盘应用走法
    show_eval(info["score_text"])       # 顺手展示评估

token = ai.request_move(board, on_ai_move)   # 发起请求，立即返回
...
ai.cancel()                              # 悔棋/开新局/关窗前调用，作废未返回的请求
```

- `callback(uci, score_cp, info)`：`info` 与 `think()` 返回结构相同（含 `pv`）。
- 返回 `token`（int）。失败时 `callback(None, 0, {"error": True})`。
- **令牌机制**：每次 `cancel()` 后，旧请求即使算完也会被丢弃，不会污染新局面。
  （底层即 `engine.Player` 的 token 实现，与原版客户端同源。）

### 5.6 `ai.set_level(level, think_ms=None)` —— 运行中换难度

```python
ai.set_level("简单")          # 菜单里做"新手/普通/大师"切换
ai.set_level("均衡", think_ms=150)   # 也可只压时间
```

### 5.7 模块级便捷函数

| 函数 | 说明 |
|---|---|
| `quick_move(fen_or_board, level="均衡") -> str|None` | 一次性走子；内部全局单例，模型只加载一次 |
| `legal_moves(fen_or_board) -> list[str]` | 当前所有合法走法（UCI 列表） |
| `game_over(fen_or_board) -> dict` | 终局判定：`{"over": bool, "result": "1-0"/"0-1"/"1/2-1/2"/None, "reason": "checkmate"/"stalemate"/"insufficient_material"/"fifty_moves"/"threefold_repetition"/"other"}` |
| `load_model(path) -> ChessNet` | 显式加载权重；`LinksAI(net=共享net)` 可让多个 AI 共用 |
| `score_text(score_cp) -> str` | 百分兵分 → `"+1.35"` / `"M5"` |
| `to_white_pov(score_cp, board) -> int` | 行棋方视角 → 白方视角 |

---

## 6. 数据约定（对接前必读）

### 6.1 走法协议：UCI
所有走法一律是 **UCI 字符串**：`起点格+终点格(+升变子)`。
`e2e4`、`g1f3`、吃子不带 x：`e4d5`、升变加一位：`e7e8q`（q/r/b/n）、
王车易位：`e1g1`（短）/ `e1c1`（长）。
你自己的棋盘只要支持"把棋子从 A 格搬到 B 格"即可直接消费。

### 6.2 输入格式
`chess.Board` 对象或 **FEN 字符串**都接受。FEN 由你的 UI 生成即可
（推荐直接用 python-chess 管理 棋规/王车易位权/过路兵/半回合钟，见第 7 节）。

### 6.3 分数视角
`score_cp` 永远是**当前行棋方视角**（AI 自己的视角）。给玩家展示评估条时：
`白方分 = to_white_pov(score_cp, board)`。

## 7. 三种集成模式（完整可抄示例）

### 模式 A：同步调用（控制台、脚本、回合制小游戏）

最小闭环三步：**求走子 → 应用 → 判终局**（完整版见 `demo_play.py`）。

```python
import chess
from engine_api import LinksAI, game_over

ai, board = LinksAI(level="均衡"), chess.Board()

while True:
    over = game_over(board)                     # ③ 终局判定
    if over["over"]:
        print(over["result"], over["reason"]); break
    if board.turn == chess.BLACK:               # AI 执黑
        uci = ai.move(board)                    # ① 求走子
        board.push_uci(uci)                     # ② 应用
    else:
        board.push_uci(input("你的走法: "))
```

### 模式 B：嵌入图形 UI（Kivy / PyQt / Tkinter / Godot 等）

要点：**主线程只做请求和画图，思考在后台线程；悔棋/新局先 `cancel()`。**

```python
class GameCtrl:
    def __init__(self):
        self.ai = LinksAI(level="强力")
        self.board = chess.Board()

    def human_moved(self, uci):                 # 玩家落子后调用
        self.board.push_uci(uci)
        if game_over(self.board)["over"]:
            self.show_result(); return
        self.ai.request_move(self.board, self.on_ai_reply)

    def on_ai_reply(self, uci, score_cp, info): # 子线程回调
        self.call_from_ui_thread(lambda: self.apply_and_redraw(uci, info["score_text"]))

    def undo(self):                             # 悔棋
        self.ai.cancel()                        # ① 先作废未完成的思考
        # ② 回退你的 board（建议整回合回退，见 FAQ）
        # ③ 回退后如果轮到 AI，再重新 request_move

    def new_game(self):
        self.ai.cancel()
        self.board = chess.Board()
```

### 模式 C：HTTP 服务（Web 前端 / 远程接入）

把引擎跑成服务，前端只需 `fetch`：

```python
# server.py —— pip install flask；运行: python server.py
from flask import Flask, request, jsonify
from engine_api import LinksAI, game_over, legal_moves

app, ai = Flask(__name__), LinksAI(level="均衡")

@app.post("/move")          # {"fen": "..."} -> {"uci": "...", "score_text": "+0.35", ...}
def move():
    fen = request.json["fen"]
    info = ai.think(fen)
    return jsonify(info)

@app.post("/state")         # 终局/合法走法查询
def state():
    fen = request.json["fen"]
    return jsonify({"over": game_over(fen), "legal": legal_moves(fen)})

app.run(port=8000)
```

```js
// 前端调用
const r = await fetch("/move", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({fen})});
const {best_move, score_text, pv} = await r.json();
```

## 8. 悔棋 / 新局 / 多开

- **棋盘状态由你持有**（`chess.Board` 或你自己的表示），AI 实例完全无状态、可随时换局。
- 悔棋：`ai.cancel()` → 回退你的棋盘 → 若轮到 AI 重新 `request_move`。
  （引擎的置换表按局面 key 索引，跨局面复用只影响速度不影响正确性。）
- 新局：`ai.cancel()` → 换新 `chess.Board()`。
- 多开：`LinksAI` 每实例独享搜索状态，可同时开多个实例（多人房间）。
  共享权重省内存：`net = load_model()` 后传给每个 `LinksAI(net=net)`。

## 9. 性能与线程安全

| 事项 | 说明 |
|---|---|
| 模型加载 | 首次 1~3 秒（只做一次；`quick_move` 自动复用单例） |
| 单步耗时 | 均衡档约 0.5~0.7s；强力档 2~3s（普通双核 CPU） |
| 线程安全 | 单实例内搜索已用锁串行；并发请多实例 |
| 内存 | 引擎 + 模型约 100~200MB |
| 平台 | 纯 CPU 推理，无 GPU 要求，Windows/Linux/macOS 均可 |

## 10. FAQ

**Q1：`import engine_api` 报 ModuleNotFoundError？**
在 `sys.path` 里加入本包目录，或把工作目录切到本包运行。`engine_api.py` 自身会把所在目录插到 `sys.path`，只要它能被 import，其余模块都能被找到。

**Q2：AI 走法如何显示"吃子/将军"特效？**
引擎只给 UCI。吃子/将军判定用 python-chess：`board.is_capture(move)`、`board.gives_check(move)`。

**Q3：想让 AI 执白怎么办？**
引擎不区分颜色：谁行棋就把当前 `board` 传给谁。开局轮到白方时直接 `request_move` 即可。

**Q4：`move()` 返回 None？**
局面已终局。先调 `game_over()`；和棋类终局（50 回合、三次重复）是否判和由你的 UI 决定，`can_claim_*` 已封装在 `game_over` 里。

**Q5：可以更强/更弱吗？**
时间换棋力：`ai.move(board, depth=8, time_ms=5000)` 单步更强；
`set_level("均衡", think_ms=100)` 更弱更快。再往上加深度收益递减，CPU 也会更慢。

**Q6：换回完整版（带界面/训练）？**
本包是精简核心。完整版 `Links_v2` 含 Kivy 界面、训练管线、lichess 微调脚本与验证工具。
