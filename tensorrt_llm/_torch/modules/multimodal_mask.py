"""
Triton kernel for generating multimodal attention masks.

This module provides efficient GPU-accelerated attention mask generation for
multimodal models (e.g., Gemma3) where:
- Text tokens attend causally to other tokens
- Image tokens within the same contiguous blob attend bidirectionally to each other
- Optional sliding window attention is supported
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _compute_blob_id_kernel(
    image_token_mask_ptr,
    blob_ids_ptr,
    seq_start,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute blob IDs for image tokens within a single sequence.
    Each contiguous group of image tokens gets a unique ID (starting from 1).
    Text tokens get ID 0.

    This kernel processes one sequence and assigns blob IDs sequentially.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    # Load image token mask for this position
    is_image = tl.load(image_token_mask_ptr + seq_start + offsets, mask=mask, other=False)

    # For blob detection, we need to check if previous token was also an image token
    # If current is image and previous was not image (or we're at position 0), this starts a new blob
    prev_offsets = offsets - 1
    prev_mask = (offsets > 0) & mask
    prev_is_image = tl.load(image_token_mask_ptr + seq_start + prev_offsets, mask=prev_mask, other=False)

    # A new blob starts when: current is image AND (previous is not image OR we're at position 0)
    starts_new_blob = is_image & ((offsets == 0) | ~prev_is_image)

    # Store 1 for new blob starts, 0 otherwise (will be prefix-summed later)
    tl.store(blob_ids_ptr + seq_start + offsets, starts_new_blob.to(tl.int32), mask=mask)


@triton.jit
def _multimodal_mask_kernel(
    # Input pointers
    image_token_mask_ptr,  # [total_tokens] bool
    qo_indptr_ptr,  # [num_contexts + 1] int
    mask_offsets_ptr,  # [num_contexts] int64 - cumulative mask sizes
    # Output pointer
    output_mask_ptr,  # [total_mask_elements] bool
    # Parameters
    sliding_window: tl.constexpr,
    has_sliding_window: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Generate multimodal attention mask for a batch of context requests.

    Each thread block handles a tile of the attention mask for one context request.
    The mask combines:
    1. Causal attention (col <= row)
    2. Optional sliding window (row - col < sliding_window)
    3. Bidirectional attention for image tokens in the same blob

    Grid: (num_row_blocks, num_col_blocks, num_contexts)
    """
    # Get context index and tile indices
    ctx_idx = tl.program_id(2)
    row_block_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    # Get sequence boundaries for this context
    seq_start = tl.load(qo_indptr_ptr + ctx_idx)
    seq_end = tl.load(qo_indptr_ptr + ctx_idx + 1)
    seq_len = seq_end - seq_start

    # Get mask offset for this context
    mask_offset = tl.load(mask_offsets_ptr + ctx_idx)

    # Compute row and column ranges for this tile
    row_start = row_block_idx * BLOCK_SIZE_ROW
    col_start = col_block_idx * BLOCK_SIZE_COL

    # Early exit if this tile is completely outside the sequence
    if row_start >= seq_len:
        return

    # Create row and column indices
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_ROW)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)

    # Create 2D grid of indices
    rows = row_offsets[:, None]  # [BLOCK_SIZE_ROW, 1]
    cols = col_offsets[None, :]  # [1, BLOCK_SIZE_COL]

    # Masks for valid positions
    row_mask = rows < seq_len
    col_mask = cols < seq_len
    valid_mask = row_mask & col_mask

    # 1. Causal mask: col <= row
    causal_mask = cols <= rows

    # 2. Sliding window mask (if enabled)
    if has_sliding_window:
        window_mask = (rows - cols) < sliding_window
        causal_mask = causal_mask & window_mask

    # 3. Bidirectional mask for image tokens in the same blob
    # Load image token mask for rows and columns
    row_is_image = tl.load(
        image_token_mask_ptr + seq_start + rows,
        mask=row_mask,
        other=False
    )
    col_is_image = tl.load(
        image_token_mask_ptr + seq_start + cols,
        mask=col_mask,
        other=False
    )

    # Both must be image tokens
    both_image = row_is_image & col_is_image

    # Check if they're in the same blob by scanning for blob boundaries
    # Two image tokens are in the same blob if there's no non-image token between them
    # For efficiency, we check: min(row, col) to max(row, col) has no False in image_token_mask
    # This is expensive for large ranges, so we use a simpler heuristic:
    # Check if all positions between min and max are image tokens

    # For the bidirectional check, we need to verify same-blob membership
    # We'll use a simpler approach: check a few positions between row and col
    min_pos = tl.minimum(rows, cols)
    max_pos = tl.maximum(rows, cols)

    # Simple same-blob check: for small distances, check all intermediate positions
    # For this kernel, we'll check if the immediate neighbors are consistent
    # This works because image tokens form contiguous blobs

    # Check position right after min_pos (if different from max_pos)
    check_pos = min_pos + 1
    intermediate_is_image = tl.load(
        image_token_mask_ptr + seq_start + check_pos,
        mask=(check_pos < max_pos) & (check_pos < seq_len) & valid_mask,
        other=True  # If no intermediate, assume same blob
    )

    # Also check position right before max_pos
    check_pos2 = max_pos - 1
    intermediate_is_image2 = tl.load(
        image_token_mask_ptr + seq_start + check_pos2,
        mask=(check_pos2 > min_pos) & (check_pos2 >= 0) & valid_mask,
        other=True
    )

    # Same blob if both endpoints are images and intermediates are images
    # Note: This is a heuristic that works for contiguous blobs
    same_blob = both_image & intermediate_is_image & intermediate_is_image2

    # Final mask: causal OR (same_blob for bidirectional attention)
    final_mask = causal_mask | same_blob

    # Apply validity mask
    final_mask = final_mask & valid_mask

    # Compute output offset: mask_offset + row * seq_len + col
    output_offsets = mask_offset + rows * seq_len + cols

    # Store results
    tl.store(output_mask_ptr + output_offsets, final_mask, mask=valid_mask)


@triton.jit
def _multimodal_mask_with_blob_ids_kernel(
    # Input pointers
    blob_ids_ptr,  # [total_tokens] int32 - precomputed blob IDs
    qo_indptr_ptr,  # [num_contexts + 1] int
    mask_offsets_ptr,  # [num_contexts] int64 - cumulative mask sizes
    # Output pointer
    output_mask_ptr,  # [total_mask_elements] bool
    # Parameters
    sliding_window: tl.constexpr,
    has_sliding_window: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Generate multimodal attention mask using precomputed blob IDs.

    This version uses precomputed blob IDs where:
    - blob_id == 0 means text token
    - blob_id > 0 means image token, with same ID indicating same blob

    Grid: (num_row_blocks, num_col_blocks, num_contexts)
    """
    # Get context index and tile indices
    ctx_idx = tl.program_id(2)
    row_block_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    # Get sequence boundaries for this context
    seq_start = tl.load(qo_indptr_ptr + ctx_idx)
    seq_end = tl.load(qo_indptr_ptr + ctx_idx + 1)
    seq_len = seq_end - seq_start

    # Get mask offset for this context
    mask_offset = tl.load(mask_offsets_ptr + ctx_idx)

    # Compute row and column ranges for this tile
    row_start = row_block_idx * BLOCK_SIZE_ROW
    col_start = col_block_idx * BLOCK_SIZE_COL

    # Early exit if this tile is completely outside the sequence
    if row_start >= seq_len:
        return

    # Create row and column indices
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_ROW)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)

    # Create 2D grid of indices
    rows = row_offsets[:, None]  # [BLOCK_SIZE_ROW, 1]
    cols = col_offsets[None, :]  # [1, BLOCK_SIZE_COL]

    # Masks for valid positions
    row_mask = rows < seq_len
    col_mask = cols < seq_len
    valid_mask = row_mask & col_mask

    # 1. Causal mask: col <= row
    causal_mask = cols <= rows

    # 2. Sliding window mask (if enabled)
    if has_sliding_window:
        window_mask = (rows - cols) < sliding_window
        causal_mask = causal_mask & window_mask

    # 3. Load blob IDs
    row_blob_id = tl.load(
        blob_ids_ptr + seq_start + rows,
        mask=row_mask,
        other=0
    )
    col_blob_id = tl.load(
        blob_ids_ptr + seq_start + cols,
        mask=col_mask,
        other=0
    )

    # Same blob: both have same non-zero blob ID
    same_blob = (row_blob_id == col_blob_id) & (row_blob_id > 0)

    # Final mask: causal OR same_blob
    final_mask = causal_mask | same_blob

    # Apply validity mask
    final_mask = final_mask & valid_mask

    # Compute output offset: mask_offset + row * seq_len + col
    output_offsets = mask_offset + rows * seq_len + cols

    # Store results
    tl.store(output_mask_ptr + output_offsets, final_mask, mask=valid_mask)


def compute_blob_ids(image_token_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute blob IDs for image tokens.

    Each contiguous group of image tokens gets a unique ID (starting from 1).
    Text tokens get ID 0.

    Args:
        image_token_mask: Boolean tensor of shape [total_tokens]

    Returns:
        Integer tensor of shape [total_tokens] with blob IDs
    """
    device = image_token_mask.device
    n = len(image_token_mask)

    # Create token type ids: 0 for text, 1 for image
    token_type_ids = image_token_mask.int()

    # Pad with zero at the start to detect transitions
    padded = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), token_type_ids])

    # Identify where blobs start (0->1 transitions)
    starts = (padded[1:] > padded[:-1]).int()

    # Cumulative sum gives unique blob ID
    blob_ids = torch.cumsum(starts, dim=0)

    # Mask out text positions (where token_type_ids == 0)
    blob_ids = blob_ids * token_type_ids

    return blob_ids.to(torch.int32)


def compute_mask_offsets(
    qo_indptr: torch.Tensor,
    num_contexts: int,
) -> torch.Tensor:
    """
    Compute cumulative mask offsets for each context request.

    Args:
        qo_indptr: Tensor of shape [num_contexts + 1] with sequence boundaries
        num_contexts: Number of context requests

    Returns:
        Tensor of shape [num_contexts] with cumulative mask element counts
    """
    # Compute sequence lengths
    seq_lens = qo_indptr[1:num_contexts + 1] - qo_indptr[:num_contexts]

    # Mask sizes are seq_len^2
    mask_sizes = seq_lens * seq_lens

    # Cumulative sum with 0 prefix
    offsets = torch.zeros(num_contexts, dtype=torch.int64, device=qo_indptr.device)
    if num_contexts > 1:
        offsets[1:] = torch.cumsum(mask_sizes[:-1], dim=0)

    return offsets


def generate_multimodal_attention_mask(
    image_token_mask: torch.BoolTensor,
    qo_indptr: torch.Tensor,
    num_contexts: int,
    effective_sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate attention mask for multimodal models using Triton kernels.

    This function creates a flattened boolean attention mask where:
    - Text tokens attend causally (with optional sliding window)
    - Image tokens in the same contiguous blob attend bidirectionally

    Args:
        image_token_mask: Boolean tensor of shape [total_tokens] indicating image tokens
        qo_indptr: Tensor of shape [num_contexts + 1] with sequence start/end indices
        num_contexts: Number of context requests in the batch
        effective_sliding_window: Optional sliding window size. None means no window.

    Returns:
        Flattened boolean mask of shape [sum(seq_len[i]^2 for i in range(num_contexts))]
    """
    device = image_token_mask.device

    # Compute blob IDs for same-blob detection
    blob_ids = compute_blob_ids(image_token_mask)

    # Compute mask offsets
    mask_offsets = compute_mask_offsets(qo_indptr, num_contexts)

    # Compute total mask size
    seq_lens = qo_indptr[1:num_contexts + 1] - qo_indptr[:num_contexts]
    max_seq_len = seq_lens.max().item()
    total_mask_size = (seq_lens * seq_lens).sum().item()

    # Allocate output
    output_mask = torch.zeros(total_mask_size, dtype=torch.bool, device=device)

    # Kernel configuration
    BLOCK_SIZE_ROW = 32
    BLOCK_SIZE_COL = 32

    num_row_blocks = (max_seq_len + BLOCK_SIZE_ROW - 1) // BLOCK_SIZE_ROW
    num_col_blocks = (max_seq_len + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL

    grid = (num_row_blocks, num_col_blocks, num_contexts)

    has_sliding_window = effective_sliding_window is not None
    sliding_window = effective_sliding_window if has_sliding_window else 0

    # Launch kernel
    _multimodal_mask_with_blob_ids_kernel[grid](
        blob_ids,
        qo_indptr,
        mask_offsets,
        output_mask,
        sliding_window,
        has_sliding_window,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )

    return output_mask

