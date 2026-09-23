"""Combines the tablebase, ANN policy, and failsafe layers into the single
move-selection call other drivers (lichess-bot's Homemade adapter, scripts,
tests) use. See docs/plans/06_Dagitim.md.

Decision order per move: tablebase (<=5 men, if the table files are present)
-> ANN top-k candidates filtered by the failsafe -> best remaining candidate.
"""
import chess
import torch

from encoding import board_to_tensor
from failsafe import pick_safe_move
from model import ChessTransformer, top_k_legal_moves
from tablebase import best_move as tablebase_best_move
from tablebase import open_tablebase, should_probe


def load_model(checkpoint_path: str, device: str = "cpu") -> ChessTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ChessTransformer(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


class NeuralChessEngine:
    def __init__(self, checkpoint_path: str, tablebase_dir: str | None = None, device: str = "cpu", top_k: int = 5):
        self.model = load_model(checkpoint_path, device)
        self.device = device
        self.top_k = top_k
        self.tablebase = open_tablebase(tablebase_dir) if tablebase_dir else None
        self.failsafe_triggers = 0  # rejected_count > 0 across calls -- see notify_stage5 note below

    def select_move(self, board: chess.Board) -> chess.Move:
        if self.tablebase is not None and should_probe(board):
            move = tablebase_best_move(board, self.tablebase)
            if move is not None:
                return move

        candidates = self._policy_candidates(board)
        move, rejected = pick_safe_move(board, candidates)
        if rejected > 0:
            # Stage 5 wants failsafe-trigger frequency monitored; this is the
            # hook (see docs/plans/05_Degerlendirme_ve_Test.md) -- a simple
            # counter is enough at this scale, no logging framework needed.
            self.failsafe_triggers += 1
        return move

    @torch.no_grad()
    def _policy_candidates(self, board: chess.Board) -> list[chess.Move]:
        tensor = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(self.device)
        from_logits, to_logits = self.model(tensor)
        return top_k_legal_moves(from_logits[0], to_logits[0], board, self.top_k)
