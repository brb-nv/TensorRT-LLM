# Patch to Skip RoPE Application in MLA

## Required Changes to CUDA Kernel (mlaKernels.cu)

### 1. Update `applyMLARopeAndAssignQKVKernelOptContext` (around line 247-276)

Replace:
```cpp
auto const position_id
    = helix_position_offsets ? helix_position_offsets[global_token_idx] : token_idx_in_kv_cache;
float2 const* rotary_coef_cache_buffer
    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

VecT q, k;
// ... load q and k ...

// Pack two elements into one for gptj rotary embedding.
#pragma unroll
for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
{
    GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
    GPTJEltT& k_ = reinterpret_cast<GPTJEltT*>(&k)[elt_id];

    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
    mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache);
}
```

With:
```cpp
VecT q, k;
// ... load q and k ...

// Apply RoPE only if cos_sin_cache is not nullptr
if (cos_sin_cache != nullptr)
{
    auto const position_id
        = helix_position_offsets ? helix_position_offsets[global_token_idx] : token_idx_in_kv_cache;
    float2 const* rotary_coef_cache_buffer
        = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

    // Pack two elements into one for gptj rotary embedding.
    #pragma unroll
    for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
    {
        GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
        GPTJEltT& k_ = reinterpret_cast<GPTJEltT*>(&k)[elt_id];

        float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
        mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache);
    }
}
```

### 2. Update `applyMLARopeAndAssignQKVKernelGeneration` (around line 443-471)

Replace:
```cpp
auto const position_id
    = (helix_position_offsets != nullptr ? helix_position_offsets[global_token_idx]
                                         : kv_cache_lengths[batch_idx] - seq_len + local_token_idx);
float2 const* rotary_coef_cache_buffer
    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

// ... load data ...

// Pack two elements into one for gptj rotary embedding.
#pragma unroll
for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
{
    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
}
```

With:
```cpp
// ... load data ...

// Apply RoPE only if cos_sin_cache is not nullptr
if (cos_sin_cache != nullptr)
{
    auto const position_id
        = (helix_position_offsets != nullptr ? helix_position_offsets[global_token_idx]
                                             : kv_cache_lengths[batch_idx] - seq_len + local_token_idx);
    float2 const* rotary_coef_cache_buffer
        = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

    // Pack two elements into one for gptj rotary embedding.
    #pragma unroll
    for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
    {
        GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

        float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
        data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
    }
}
```

### 3. Update `applyMLARopeAppendPagedKVAssignQKernel` (around line 751-776)

Replace:
```cpp
auto const position_id = token_idx_in_kv_cache;
float2 const* rotary_coef_cache_buffer
    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

// ... load data ...

// Pack two elements into one for gptj rotary embedding.
#pragma unroll
for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
{
    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
}
```

With:
```cpp
// ... load data ...

// Apply RoPE only if cos_sin_cache is not nullptr
if (cos_sin_cache != nullptr)
{
    auto const position_id = token_idx_in_kv_cache;
    float2 const* rotary_coef_cache_buffer
        = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

    // Pack two elements into one for gptj rotary embedding.
    #pragma unroll
    for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
    {
        GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

        float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
        data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
    }
}
```

## Summary
These changes wrap all RoPE application code in `if (cos_sin_cache != nullptr)` checks, allowing the kernels to skip RoPE when `None` is passed from Python.

