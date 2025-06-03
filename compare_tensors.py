import argparse
import numpy as np
import os

tensornames = ["input_embeddings_scaled", "attention_input_0", "attention_output_0", "attention_output_post_layernorm_0", "attention_output_with_residual_0", "mlp_input_0", "mlp_output_0", "mlp_output_post_layernorm_0", "mlp_output_with_residual_0", "final_norm_output"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trt_debug",
        type=str,
        default="/home/bbuddharaju/scratch/TensorRT-LLM/trtllm_debug/",
    )

    parser.add_argument(
        "--hf_debug",
        type=str,
        default="/home/bbuddharaju/scratch/TensorRT-LLM/hf_debug/",
    )

    args = parser.parse_args()
    assert os.path.isdir(args.trt_debug)
    assert os.path.isdir(args.hf_debug)

    for tensorname in tensornames:
        trt_filepath = os.path.join(args.trt_debug, tensorname + ".npy")
        hf_filepath = os.path.join(args.hf_debug, tensorname + ".npy")
        trt_tensor = np.load(trt_filepath)
        hf_tensor = np.load(hf_filepath)
        diff = np.abs(trt_tensor - hf_tensor)
        if trt_tensor.shape != hf_tensor.shape:
            trt_tensor = np.expand_dims(trt_tensor, axis=0)
            assert trt_tensor.shape == hf_tensor.shape, f"Tensor {tensorname} has shape mismatch. {trt_tensor.shape}, {hf_tensor.shape}."
        print(f"Tensor: {tensorname}")
        print(f"  Max difference: {np.max(diff):.6f}")
        print(f"  Mean difference: {np.mean(diff):.6f}")


if __name__ == "__main__":
    main()
