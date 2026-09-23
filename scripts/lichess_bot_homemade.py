"""Drop-in addition to lichess-bot's homemade.py (a separate git clone --
see docs/reference/Teknoloji_Yigini_ve_Kaynaklar.md). This file can't be run
or tested from THIS repo: it imports lib.engine_wrapper/lib.lichess_types,
which only exist inside a lichess-bot checkout. The logic it calls into
(NeuralChessEngine) is the part that's actually tested, in tests/test_engine.py.

Wiring it in:
1. Make this project's src/ importable from inside the lichess-bot checkout,
   either by adding it to PYTHONPATH or inserting sys.path here, then copy
   this class into (or import it from) lichess-bot's own homemade.py.
2. In lichess-bot's config.yml:
       engine:
         name: "NeuralChess"
         protocol: "homemade"
3. Set NEURAL_CHESS_CHECKPOINT (and optionally NEURAL_CHESS_TABLEBASE_DIR)
   as environment variables before starting lichess-bot.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
from chess.engine import PlayResult
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import HOMEMADE_ARGS_TYPE

from engine import NeuralChessEngine

CHECKPOINT_PATH = os.environ.get("NEURAL_CHESS_CHECKPOINT", "checkpoints/run2/best.pt")
TABLEBASE_DIR = os.environ.get("NEURAL_CHESS_TABLEBASE_DIR")  # unset = ANN + failsafe only, no tablebase


class NeuralChess(MinimalEngine):
    """lichess-bot Homemade adapter for this project's tablebase + ANN + failsafe engine."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._engine = NeuralChessEngine(CHECKPOINT_PATH, tablebase_dir=TABLEBASE_DIR)

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        return PlayResult(self._engine.select_move(board), None)
