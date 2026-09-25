"""Encoder-only transformer policy network: board -> (from-square, to-square) logits.

Consumes exactly the (22,8,8) board tensors produced by encoding.board_to_tensor
(see build_dataset.py for the shard format). Each of the 64 squares becomes one
token (piece type+color embedding + learned per-square positional embedding +
a mobility-flag embedding, from encoding.py's channel 18); side-to-move, the
four castling rights, en-passant, and the previous move's from/to squares
become eight extra tokens (a shared "flag" embedding for 0/1 plus a
per-token-kind embedding; the en-passant and last-move tokens also reuse the
square positional embedding for their target square). A Geometric Attention
Bias (GAB, run6) reads the raw board tensor and produces a dynamic per-head
64x64 additive bias fed into every encoder layer's self-attention (via
nn.TransformerEncoder's mask= argument), on top of the fixed token embeddings.
Two linear heads turn each of the 64 board-token outputs into one from-square
and one to-square logit.
"""
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import NUM_CHANNELS

# Confirmed bug (torch 2.14.0+cpu): nn.TransformerEncoder's fused eval+no_grad
# ("fast path") kernel silently produces NaN output when given a float
# attn_mask (our GAB bias) -- the plain/slow path (training mode, or any mode
# with grad tracking) is correct. Since inference (engine.py) and validation
# (train.py) both run under eval()+no_grad(), this must be disabled globally
# or every masked forward pass in eval mode returns garbage. See
# tests/test_model.py::test_encoder_mask_survives_eval_and_no_grad.
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)

NUM_SQUARES = 64
NUM_PIECE_CLASSES = 13  # 0 = empty, 1-6 = white P N B R Q K, 7-12 = black P N B R Q K
NUM_EXTRA_TOKENS = 8  # side-to-move, castle WK/WQ/BK/BQ, en-passant, last-move-from, last-move-to


class GeometricAttentionBias(nn.Module):
    """Board-state-conditioned additive attention bias (Chessformer-style
    GAB, arXiv:2605.19091). Compresses the whole raw board tensor to a small
    per-head "template mixture", then expands it through a single 64x64
    projection SHARED across all heads (broadcast, not one projection per
    head) -- keeping the added parameter count small (~349K at the
    defaults) instead of ~4M+ for a per-head-dense version.
    """

    def __init__(self, in_channels: int, nhead: int, d_compress: int = 128, d_templates: int = 32):
        super().__init__()
        self.nhead = nhead
        self.d_templates = d_templates
        self.compress_fc = nn.Linear(in_channels * NUM_SQUARES, d_compress)
        self.head_proj = nn.Linear(d_compress, nhead * d_templates)
        self.template_bank = nn.Linear(d_templates, NUM_SQUARES * NUM_SQUARES)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """boards: (B, in_channels, 8, 8) float -> (B, nhead, 64, 64)."""
        batch = boards.shape[0]
        z = F.relu(self.compress_fc(boards.reshape(batch, -1)))
        w = self.head_proj(z).reshape(batch, self.nhead, self.d_templates)
        bias = self.template_bank(w)  # (B, nhead, 4096), template_bank shared across heads
        return bias.reshape(batch, self.nhead, NUM_SQUARES, NUM_SQUARES)


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
        self.nhead = nhead
        self.piece_embed = nn.Embedding(NUM_PIECE_CLASSES, d_model)
        self.square_pos_embed = nn.Embedding(NUM_SQUARES, d_model)
        self.extra_type_embed = nn.Embedding(NUM_EXTRA_TOKENS, d_model)
        self.flag_embed = nn.Embedding(2, d_model)
        self.mobility_embed = nn.Embedding(2, d_model)
        self.gab = GeometricAttentionBias(in_channels=NUM_CHANNELS, nhead=nhead)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.from_head = nn.Linear(d_model, 1)
        self.to_head = nn.Linear(d_model, 1)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """boards: (B, NUM_CHANNELS, 8, 8) -> (from_logits, to_logits), each (B, 64)."""
        boards = boards.float()
        batch = boards.shape[0]
        device = boards.device

        piece_planes = boards[:, 0:12]  # (B, 12, 8, 8)
        occupied = piece_planes.sum(dim=1) > 0
        piece_idx = piece_planes.argmax(dim=1) + 1
        piece_idx = torch.where(occupied, piece_idx, torch.zeros_like(piece_idx))
        piece_idx = piece_idx.reshape(batch, NUM_SQUARES).long()

        square_ids = torch.arange(NUM_SQUARES, device=device)
        mobility = boards[:, 18].reshape(batch, NUM_SQUARES).long()
        board_tok = (
            self.piece_embed(piece_idx)
            + self.square_pos_embed(square_ids).unsqueeze(0)
            + self.mobility_embed(mobility)
        )

        stm = boards[:, 12, 0, 0].long()
        castle_wk = boards[:, 13, 0, 0].long()
        castle_wq = boards[:, 14, 0, 0].long()
        castle_bk = boards[:, 15, 0, 0].long()
        castle_bq = boards[:, 16, 0, 0].long()
        ep_plane = boards[:, 17].reshape(batch, NUM_SQUARES)
        ep_present = (ep_plane.sum(dim=1) > 0).long()
        ep_square = ep_plane.argmax(dim=1)
        last_from_plane = boards[:, 19].reshape(batch, NUM_SQUARES)
        last_from_present = (last_from_plane.sum(dim=1) > 0).long()
        last_from_square = last_from_plane.argmax(dim=1)
        last_to_plane = boards[:, 20].reshape(batch, NUM_SQUARES)
        last_to_present = (last_to_plane.sum(dim=1) > 0).long()
        last_to_square = last_to_plane.argmax(dim=1)

        type_ids = torch.arange(NUM_EXTRA_TOKENS, device=device)
        type_embeds = self.extra_type_embed(type_ids)  # (8, d_model)

        flags = torch.stack([stm, castle_wk, castle_wq, castle_bk, castle_bq], dim=1)  # (B, 5)
        plain_tok = type_embeds[:5].unsqueeze(0) + self.flag_embed(flags)  # (B, 5, d_model)
        ep_tok = (
            type_embeds[5].unsqueeze(0)
            + self.flag_embed(ep_present)
            + self.square_pos_embed(ep_square)
        )  # (B, d_model)
        last_from_tok = (
            type_embeds[6].unsqueeze(0)
            + self.flag_embed(last_from_present)
            + self.square_pos_embed(last_from_square)
        )  # (B, d_model)
        last_to_tok = (
            type_embeds[7].unsqueeze(0)
            + self.flag_embed(last_to_present)
            + self.square_pos_embed(last_to_square)
        )  # (B, d_model)
        extra_tok = torch.cat(
            [plain_tok, ep_tok.unsqueeze(1), last_from_tok.unsqueeze(1), last_to_tok.unsqueeze(1)], dim=1
        )  # (B, 8, d_model)

        tokens = torch.cat([board_tok, extra_tok], dim=1)  # (B, 72, d_model)
        total_tokens = NUM_SQUARES + NUM_EXTRA_TOKENS

        bias64 = self.gab(boards)  # (B, nhead, 64, 64)
        full_bias = boards.new_zeros(batch, self.nhead, total_tokens, total_tokens)
        full_bias[:, :, :NUM_SQUARES, :NUM_SQUARES] = bias64  # extra tokens get no bias
        mask = full_bias.reshape(batch * self.nhead, total_tokens, total_tokens)

        encoded = self.encoder(tokens, mask=mask)
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


def _prefer_queen(candidates: list[chess.Move]) -> chess.Move:
    """Among legal moves sharing a (from, to) pair (underpromotion choices),
    prefer queen promotion; otherwise just the one candidate."""
    for m in candidates:
        if m.promotion in (None, chess.QUEEN):
            return m
    return candidates[0]


def top_k_legal_moves(from_logits: torch.Tensor, to_logits: torch.Tensor, board: chess.Board, k: int) -> list[chess.Move]:
    """Up to k distinct legal moves by joint from/to score, most likely
    first. Promotion ties collapse to one entry each (see _prefer_queen) so
    the ranking reflects distinct (from, to) squares, not raw score-matrix
    cells."""
    joint = mask_move_logits(from_logits, to_logits, legal_move_mask(board))
    order = torch.argsort(joint.flatten(), descending=True)
    moves: list[chess.Move] = []
    seen_squares: set[tuple[int, int]] = set()
    for idx in order.tolist():
        if joint.flatten()[idx].item() == float("-inf") or len(moves) >= k:
            break
        from_sq, to_sq = divmod(idx, NUM_SQUARES)
        if (from_sq, to_sq) in seen_squares:
            continue
        seen_squares.add((from_sq, to_sq))
        candidates = [m for m in board.legal_moves if m.from_square == from_sq and m.to_square == to_sq]
        moves.append(_prefer_queen(candidates))
    return moves


def select_legal_move(from_logits: torch.Tensor, to_logits: torch.Tensor, board: chess.Board) -> chess.Move:
    """Highest-scoring legal move. When a (from, to) pair matches several
    legal moves (underpromotion choices), queen promotion is preferred."""
    return top_k_legal_moves(from_logits, to_logits, board, k=1)[0]
