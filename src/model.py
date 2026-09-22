"""Encoder-only transformer policy network: board -> (from-square, to-square) logits.

Consumes exactly the (18,8,8) board tensors produced by encoding.board_to_tensor
(see build_dataset.py for the shard format). Each of the 64 squares becomes one
token (piece type+color embedding + learned per-square positional embedding);
side-to-move, the four castling rights, and en-passant become six extra tokens
(a shared "flag" embedding for 0/1 plus a per-token-kind embedding; the
en-passant token also reuses the square positional embedding for its target
square). Two linear heads turn each of the 64 board-token outputs into one
from-square and one to-square logit.
"""
import chess
import numpy as np
import torch
import torch.nn as nn

NUM_SQUARES = 64
NUM_PIECE_CLASSES = 13  # 0 = empty, 1-6 = white P N B R Q K, 7-12 = black P N B R Q K
NUM_EXTRA_TOKENS = 6  # side-to-move, castle WK/WQ/BK/BQ, en-passant


class ChessTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.piece_embed = nn.Embedding(NUM_PIECE_CLASSES, d_model)
        self.square_pos_embed = nn.Embedding(NUM_SQUARES, d_model)
        self.extra_type_embed = nn.Embedding(NUM_EXTRA_TOKENS, d_model)
        self.flag_embed = nn.Embedding(2, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.from_head = nn.Linear(d_model, 1)
        self.to_head = nn.Linear(d_model, 1)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """boards: (B, 18, 8, 8) -> (from_logits, to_logits), each (B, 64)."""
        boards = boards.float()
        batch = boards.shape[0]
        device = boards.device

        piece_planes = boards[:, 0:12]  # (B, 12, 8, 8)
        occupied = piece_planes.sum(dim=1) > 0
        piece_idx = piece_planes.argmax(dim=1) + 1
        piece_idx = torch.where(occupied, piece_idx, torch.zeros_like(piece_idx))
        piece_idx = piece_idx.reshape(batch, NUM_SQUARES).long()

        square_ids = torch.arange(NUM_SQUARES, device=device)
        board_tok = self.piece_embed(piece_idx) + self.square_pos_embed(square_ids).unsqueeze(0)

        stm = boards[:, 12, 0, 0].long()
        castle_wk = boards[:, 13, 0, 0].long()
        castle_wq = boards[:, 14, 0, 0].long()
        castle_bk = boards[:, 15, 0, 0].long()
        castle_bq = boards[:, 16, 0, 0].long()
        ep_plane = boards[:, 17].reshape(batch, NUM_SQUARES)
        ep_present = (ep_plane.sum(dim=1) > 0).long()
        ep_square = ep_plane.argmax(dim=1)

        type_ids = torch.arange(NUM_EXTRA_TOKENS, device=device)
        type_embeds = self.extra_type_embed(type_ids)  # (6, d_model)

        flags = torch.stack([stm, castle_wk, castle_wq, castle_bk, castle_bq], dim=1)  # (B, 5)
        plain_tok = type_embeds[:5].unsqueeze(0) + self.flag_embed(flags)  # (B, 5, d_model)
        ep_tok = (
            type_embeds[5].unsqueeze(0)
            + self.flag_embed(ep_present)
            + self.square_pos_embed(ep_square)
        )  # (B, d_model)
        extra_tok = torch.cat([plain_tok, ep_tok.unsqueeze(1)], dim=1)  # (B, 6, d_model)

        tokens = torch.cat([board_tok, extra_tok], dim=1)  # (B, 70, d_model)
        encoded = self.encoder(tokens)
        board_encoded = encoded[:, :NUM_SQUARES]  # (B, 64, d_model)

        from_logits = self.from_head(board_encoded).squeeze(-1)
        to_logits = self.to_head(board_encoded).squeeze(-1)
        return from_logits, to_logits


# --- Legal-move masking (inference time) ---------------------------------

def legal_move_mask(board: chess.Board) -> np.ndarray:
    """(64, 64) bool array, [from, to] True iff some legal move uses that
    from/to square pair (promotion piece not distinguished)."""
    mask = np.zeros((NUM_SQUARES, NUM_SQUARES), dtype=bool)
    for move in board.legal_moves:
        mask[move.from_square, move.to_square] = True
    return mask


def mask_move_logits(from_logits: torch.Tensor, to_logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """(64,) + (64,) -> (64, 64) joint from/to score matrix with illegal
    (from, to) pairs set to -inf, ready for argmax/sampling/topk."""
    joint = from_logits.unsqueeze(-1) + to_logits.unsqueeze(-2)
    mask_t = torch.from_numpy(mask).to(joint.device)
    return joint.masked_fill(~mask_t, float("-inf"))


def select_legal_move(from_logits: torch.Tensor, to_logits: torch.Tensor, board: chess.Board) -> chess.Move:
    """Highest-scoring legal move. When a (from, to) pair matches several
    legal moves (underpromotion choices), queen promotion is preferred."""
    joint = mask_move_logits(from_logits, to_logits, legal_move_mask(board))
    from_sq, to_sq = divmod(int(joint.argmax().item()), NUM_SQUARES)
    candidates = [m for m in board.legal_moves if m.from_square == from_sq and m.to_square == to_sq]
    for m in candidates:
        if m.promotion in (None, chess.QUEEN):
            return m
    return candidates[0]
