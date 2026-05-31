import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    generated_tokens = []
    past_key_values = None

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token_id)

        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token_id)

    return [t.item() for t in generated_tokens]


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    model = build_model(torch.float16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. KV cache (use_cache=True + feed only last token): ~3-4x
#    The baseline recomputes attention over the full growing sequence every step,
#    making it O(n^2). With KV cache, each decode step only processes 1 token
#    and reuses cached keys/values from prior steps.
#
# 2. Remove .item() sync per step: ~1.3x
#    .item() forces a CPU-GPU synchronization, stalling the GPU pipeline.
#    Instead, keep tokens as tensors and collect values once at the end.
#
# 3. FP16 dtype: ~1.3-1.5x
#    Halves memory traffic for all weight loads, making decode steps
#    (which are memory-bandwidth-bound) proportionally faster.
#
# 4. torch.no_grad(): ~1.1x
#    Disables autograd graph construction, reducing CPU overhead and memory.
#
# 5. Eliminated torch.cat reallocation (subsumed by KV cache fix):
#    No longer grow the input sequence — just pass the single new token.
#
# Biggest impact and why:
#
# KV cache was by far the largest win (~3-4x). Without it, step N recomputes
# attention over all N prior tokens, making total work O(n^2) in sequence
# length. With it, each step does O(1) new attention work (one query against
# cached K/V), reducing total generation to O(n). For 128 decode steps from
# a 1024-token prompt, this eliminates the vast majority of redundant compute.
#