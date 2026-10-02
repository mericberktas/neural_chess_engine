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
With gab_per_layer (run9) each encoder layer gets its own bias instead of all
sharing one. Two linear heads turn each of the 64 board-token outputs into one from-square
and one to-square logit.
"""
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import NUM_CHANNELS, SEE_CHANNEL

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
# attack-graph relation classes: attacker type (6) x target (6 piece types + empty = 7) x same-colour (2)
NUM_ATTACK_RELATIONS = 6 * 7 * 2
NUM_EXTRA_TOKENS = 8  # side-to-move, castle WK/WQ/BK/BQ, en-passant, last-move-from, last-move-to


class GeometricAttentionBias(nn.Module):
    """Board-state-conditioned additive attention bias (Chessformer-style
    GAB, arXiv:2605.19091). Compresses the whole raw board tensor to a small
    per-head "template mixture", then expands it through a single 64x64
    projection SHARED across all heads (broadcast, not one projection per
    head) -- keeping the added parameter count small (~349K at the
    defaults) instead of ~4M+ for a per-head-dense version.
    """

    def __init__(self, in_channels: int, nhead: int, d_compress: int = 128, d_templates: int = 32, num_outputs: int = 1):
        super().__init__()
        self.nhead = nhead
        self.d_templates = d_templates
        self.num_outputs = num_outputs  # independent biases (one per encoder layer when > 1); compress_fc/template_bank stay shared
        self.compress_fc = nn.Linear(in_channels * NUM_SQUARES, d_compress)
        self.head_proj = nn.Linear(d_compress, num_outputs * nhead * d_templates)
        self.template_bank = nn.Linear(d_templates, NUM_SQUARES * NUM_SQUARES)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """boards: (B, in_channels, 8, 8) float -> (B, nhead, 64, 64), or
        (B, num_outputs, nhead, 64, 64) when num_outputs > 1."""
        batch = boards.shape[0]
        z = F.relu(self.compress_fc(boards.reshape(batch, -1)))
        w = self.head_proj(z).reshape(batch, self.num_outputs * self.nhead, self.d_templates)
        bias = self.template_bank(w)  # (B, num_outputs*nhead, 4096), template_bank shared across heads/outputs
        if self.num_outputs == 1:
            return bias.reshape(batch, self.nhead, NUM_SQUARES, NUM_SQUARES)
        return bias.reshape(batch, self.num_outputs, self.nhead, NUM_SQUARES, NUM_SQUARES)


def unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    """(..., K) uint8 -> (..., K*8) bool, little-endian bit order within each
    byte -- the inverse of np.packbits(..., bitorder="little"), which is how
    encoding.position_extras stores its legal mask and attack bitboards."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*packed.shape[:-1], -1).bool()


class ChessTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        gab_per_layer: bool = False,
        attack_bias: bool = False,
        see_channel: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.gab_per_layer = gab_per_layer
        self.attack_bias = attack_bias
        self.see_channel = see_channel
        keep = torch.ones(1, NUM_CHANNELS, 1, 1)
        keep[:, SEE_CHANNEL] = 0
        self.register_buffer("channel_keep", keep, persistent=False)  # applied only when see_channel is False
        if attack_bias:
            # [layer, 0=attacker->target, 1=target<-attacker, relation, head]; zero-init so training starts from the plain model
            self.attack_embed = nn.Parameter(torch.zeros(num_layers, 2, NUM_ATTACK_RELATIONS, nhead))
        self.piece_embed = nn.Embedding(NUM_PIECE_CLASSES, d_model)
        self.square_pos_embed = nn.Embedding(NUM_SQUARES, d_model)
        self.extra_type_embed = nn.Embedding(NUM_EXTRA_TOKENS, d_model)
        self.flag_embed = nn.Embedding(2, d_model)
        self.mobility_embed = nn.Embedding(2, d_model)
        self.gab = GeometricAttentionBias(
            in_channels=NUM_CHANNELS, nhead=nhead, num_outputs=num_layers if gab_per_layer else 1,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.from_head = nn.Linear(d_model, 1)
        self.to_head = nn.Linear(d_model, 1)

    @staticmethod
    def _piece_indices(boards: torch.Tensor) -> torch.Tensor:
        """(B, C, 8, 8) -> (B, 64) long: 0 = empty, 1-6 white P N B R Q K, 7-12 black."""
        batch = boards.shape[0]
        piece_planes = boards[:, 0:12]  # (B, 12, 8, 8)
        occupied = piece_planes.sum(dim=1) > 0
        piece_idx = piece_planes.argmax(dim=1) + 1
        piece_idx = torch.where(occupied, piece_idx, torch.zeros_like(piece_idx))
        return piece_idx.reshape(batch, NUM_SQUARES).long()

    @staticmethod
    def _attack_relations(piece_idx: torch.Tensor, attacks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,64) piece indices + (B,64,8) packed attack bitboards ->
        (rel (B,64,64) long, mask (B,64,64) bool), both indexed [from, to]:
        mask is "the piece on `from` attacks/defends `to`", rel its class --
        (attacker type * 7 + target type, 6 = empty target) * 2 + target is
        occupied by the same colour as the attacker (a defence, not a capture)."""
        occupied = piece_idx > 0
        base = (piece_idx - 1).clamp(min=0)
        piece_type, colour = base % 6, base // 6
        target_class = torch.where(occupied, piece_type, torch.full_like(piece_type, 6))
        same = (colour.unsqueeze(2) == colour.unsqueeze(1)) & occupied.unsqueeze(1)
        rel = (piece_type.unsqueeze(2) * 7 + target_class.unsqueeze(1)) * 2 + same.long()
        return rel, unpack_bits(attacks.to(piece_idx.device))

    def _attack_term(self, rel: torch.Tensor, mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """(B, nhead, 64, 64) additive attention bias for one layer: the
        learned weight of the (from -> to) attack relation, plus the reverse
        weight for the (to -> from) relation so a square can also attend to
        whatever attacks or defends it."""
        forward = self.attack_embed[layer_idx, 0][rel]  # (B, 64, 64, H)
        backward = self.attack_embed[layer_idx, 1][rel.transpose(1, 2)]
        term = mask.unsqueeze(-1) * forward + mask.transpose(1, 2).unsqueeze(-1) * backward
        return term.permute(0, 3, 1, 2)

    def encode(self, boards: torch.Tensor, attacks: torch.Tensor | None = None) -> torch.Tensor:
        """boards: (B, NUM_CHANNELS, 8, 8) -> (B, 64, d_model) board-token features.
        attacks: (B, 64, 8) uint8 attack bitboards, required iff attack_bias."""
        boards = boards.float()
        if not self.see_channel:
            boards = boards * self.channel_keep
        batch = boards.shape[0]
        device = boards.device

        piece_idx = self._piece_indices(boards)
        if self.attack_bias:
            if attacks is None:
                raise ValueError("attack_bias model needs the attacks array (shard format v2)")
            attack_rel, attack_mask = self._attack_relations(piece_idx, attacks)

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

        def layer_mask(bias64: torch.Tensor) -> torch.Tensor:
            full_bias = boards.new_zeros(batch, self.nhead, total_tokens, total_tokens)
            full_bias[:, :, :NUM_SQUARES, :NUM_SQUARES] = bias64  # extra tokens get no bias
            return full_bias.reshape(batch * self.nhead, total_tokens, total_tokens)

        gab_bias = self.gab(boards)  # (B, nhead, 64, 64), or (B, num_layers, nhead, 64, 64) if gab_per_layer

        encoded = tokens
        for i, layer in enumerate(self.encoder.layers):
            bias_i = gab_bias[:, i] if self.gab_per_layer else gab_bias
            if self.attack_bias:
                bias_i = bias_i + self._attack_term(attack_rel, attack_mask, i)
            encoded = layer(encoded, src_mask=layer_mask(bias_i))
        if self.encoder.norm is not None:
            encoded = self.encoder.norm(encoded)
        return encoded[:, :NUM_SQUARES]  # (B, 64, d_model)

    def forward(self, boards: torch.Tensor, attacks: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """boards: (B, NUM_CHANNELS, 8, 8) -> (from_logits, to_logits), each (B, 64)."""
        board_encoded = self.encode(boards, attacks)
        return self.from_head(board_encoded).squeeze(-1), self.to_head(board_encoded).squeeze(-1)

    def joint_logits(self, boards: torch.Tensor, attacks: torch.Tensor | None = None) -> torch.Tensor:
        """(B, 64, 64) score for every (from, to) pair -- the quantity the
        loss/eval/inference rank moves by (additive here: from + to)."""
        from_logits, to_logits = self(boards, attacks)
        return from_logits.unsqueeze(-1) + to_logits.unsqueeze(-2)


# --- Legal-move masking (inference time) ---------------------------------

def legal_move_mask(board: chess.Board) -> np.ndarray:
    """(64, 64) bool array, [from, to] True iff some legal move uses that
    from/to square pair (promotion piece not distinguished)."""
    mask = np.zeros((NUM_SQUARES, NUM_SQUARES), dtype=bool)
    for move in board.legal_moves:
        mask[move.from_square, move.to_square] = True
    return mask


def mask_joint_logits(joint: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """(64, 64) joint score matrix with illegal (from, to) pairs set to -inf."""
    mask_t = torch.from_numpy(mask).to(joint.device)
    return joint.masked_fill(~mask_t, float("-inf"))


def mask_move_logits(from_logits: torch.Tensor, to_logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """(64,) + (64,) -> (64, 64) joint from/to score matrix with illegal
    (from, to) pairs set to -inf, ready for argmax/sampling/topk."""
    return mask_joint_logits(from_logits.unsqueeze(-1) + to_logits.unsqueeze(-2), mask)


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
    return top_k_legal_moves_from_joint(from_logits.unsqueeze(-1) + to_logits.unsqueeze(-2), board, k)


def top_k_legal_moves_from_joint(joint: torch.Tensor, board: chess.Board, k: int) -> list[chess.Move]:
    """Same as top_k_legal_moves, for a model that scores (from, to) pairs
    directly: joint is its un-masked (64, 64) score matrix."""
    joint = mask_joint_logits(joint, legal_move_mask(board))
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
