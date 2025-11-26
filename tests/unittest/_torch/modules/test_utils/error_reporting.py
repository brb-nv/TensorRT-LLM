# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for error reporting in MLA tests."""

import torch


def report_tensor_diff(output, ref_output, atol, rtol, prefix=""):
    """Compare two tensors and report detailed error statistics.
    
    Args:
        output: Test output tensor
        ref_output: Reference output tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
        prefix: Prefix string for error messages
        
    Returns:
        Number of mismatched elements
    """
    err = torch.abs(output - ref_output)
    ref_abs = torch.abs(ref_output)
    ref_abs[ref_abs == 0] = torch.finfo(ref_abs.dtype).smallest_normal
    rel_err = err / ref_abs
    
    # Find maximum error and its index
    max_err_idx = torch.unravel_index(torch.argmax(err - atol - rtol * ref_abs), err.shape)
    values_err = (output[max_err_idx].item(), ref_output[max_err_idx].item())
    
    # Find maximum absolute error
    max_abs_err_idx = torch.unravel_index(torch.argmax(err), err.shape)
    values_abs = (output[max_abs_err_idx].item(), ref_output[max_abs_err_idx].item())
    
    # Find maximum relative error
    max_rel_err_idx = torch.unravel_index(torch.argmax(rel_err), rel_err.shape)
    values_rel = (output[max_rel_err_idx].item(), ref_output[max_rel_err_idx].item())
    
    max_abs_err = err[max_abs_err_idx].item()
    max_rel_err = rel_err[max_rel_err_idx].item()
    
    # Convert to lists for printing
    max_err_idx = [x.item() for x in max_err_idx]
    max_abs_err_idx = [x.item() for x in max_abs_err_idx]
    max_rel_err_idx = [x.item() for x in max_rel_err_idx]
    
    # Count errors
    isclose = err < atol + rtol * ref_abs
    n_error = (~isclose).sum().item()
    
    print(
        f"{prefix}: {n_error} errors, max error index: {max_err_idx} "
        f"(test/ref values: {values_err}), max abs error index: {max_abs_err_idx} "
        f"(test/ref values: {values_abs}, err: {max_abs_err}), max rel error index: {max_rel_err_idx} "
        f"(test/ref values: {values_rel}, err: {max_rel_err}), atol: {atol}, rtol: {rtol}"
    )
    return n_error


