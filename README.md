# MegaFold: System-Level Optimizations for Accelerating Protein Structure Prediction Models

## About 

[MegaFold](TODO:add arxiv link) is a cross-platform system to accelerate protein structure prediction models (e.g., AlphaFold3, AlphaFold2).

Why MegaFold? 

- **Cross-platform support**: Supports execution on heterogeneous devices, including NVIDIA GPUs, AMD GPUs, and CPUs.
- **Speed improvement**: Accelerates per-iteration training time by up to 1.73x
- **Memory reduction**: Reduces peak memory during training by up to 1.23x
- **Sequence length extension**: Enables training on 1.35x longer sequence lengths


## Usage

### alphafold3-pytorch
The `alphafold3-pytorch` folder includes AF3 training code (baseline and end-to-end MegaFold integrations) and instructions to reproduce our paper results. More details in `alphafold3-pytorch/README.md`.

---
### Data-loading optimizations
The file `alphafold3-pytorch/omnifold/inputs.py` includes the data pipeline and implementation details for the ahead-of-time cache-based data loading optimizations. 

You can find details on deterministic input features cache in lines 4536-4553 and on MSA features cache in lines 4670-4732.

---
### FusedEvoAttention
The folder `FusedEvoAttention` includes source code of FusedEvoAttention kernel. 

<details>
<summary>Expand for step-by-step guide</summary>

#### Step 1: Import

```
from evoformer import TritonEvoformer
```

#### Step 2: In-code usage

`FusedEvoAttention` supports 4 main types of EvoAttention in AlphaFold models, shown in the below examples. For accuracy, you need to adjust your inputs to their suggested shapes before passing in. Acronyms: `N_seq` is the MSA depth; `N_res` is the input sequence length. 

**a. Single Attention with Pair Bias**

```
# Q, K, V:     [Batch, 1, N_res, Head, Dim]
# mask:        [Batch, 1, 1, 1, N_res]
# pair_bias:   [Batch, 1, Head, N_res, N_res]
out = TritonEvoformer(Q, K, V, mask, pair_bias)
```

**b. Triangle Attention (around starting node and around ending node)**

```
# Q, K, V:     [Batch, N_res, N_res, Head, Dim]
# mask:        [Batch, N_res, 1, 1, N_res]
# pair_bias:   [Batch, 1, Head, N_res, N_res]
out = TritonEvoformer(Q, K, V, mask, pair_bias)
```

**c. MSA Row-wise Attention**

```
# Q, K, V:     [Batch, N_seq, N_res, Head, Dim]
# mask:        [Batch, N_seq, 1, 1, N_res]
# pair_bias:   [Batch, 1, Head, N_res, N_res]
out = TritonEvoformer(Q, K, V, mask, pair_bias)
```

**d. MSA Column-wise Attention**

```
# Q, K, V:     [Batch, N_res, N_seq, Head, Dim]
# mask:        [Batch, N_seq, 1, 1, N_res]
out = TritonEvoformer(Q, K, V, mask)
```


#### Step 3: Autotuning for optimal performance

To achieve peak performance, the kernel's configuration (block sizes, num warps, etc.) should be tuned to your specific hardware and input shapes.

1. Import `TritonEvoformer` as shown above.
2. Use it in your model's training or inference script.
3. Run your script with autotuning enabled: 

```
TRITON_PRINT_AUTOTUNING=1 python your_script.py
```

4. With autotuning enabled, Triton will explore multiple kernel configurations. Then, it will print the best configuration for your input.
5. Let the script run for several training iterations. Take note of the most frequently selected configuration—it is likely the best one for your target hardware and input shapes (sequence length).
6. Manually write in the best configurations for each JIT kernels and comment out the `@triton.autotune` decorator of each jit kernels. An example of an autotuned kernel for NVIDIA H200 and sequence length 384 is provided in `autotuned.py`.
7. Use the modified kernel in your real workloads for best performance.

</details>

---
### FusedLayernormLinear
The folder `FusedLayernormLinear` includes source code of fused layernorm-linear kernel. 

<details>
<summary>Expand for step-by-step guide</summary>

#### Step 1: Import

```
from fused_layernorm_linear import LayernormLinear
```

#### Step 2: In-code usage

FusedLayernormLinear fuses sequential `LayerNorm` and `Linear` layers. You can replace any such occurences with `LayernormLinear`.

```diff
# init
- layernorm = LayerNorm(dim_K)
- linear = Linear(dim_K, dim_N)
+ fused_layernorm_linear = LayernormLinear(dim_K, dim_N)

# model pass
- layernorm_linear_out = linear(layernorm(input))
+ layernorm_linear_out = fused_layernorm_linear(input)
```

- **[AMD users]**: Use `helper_amd.py` instead of `helper.py`: <code>from ~~helper~~ helper_amd import calculate_config_layernorm_linear </code>

- **NOTE**: `LayernormLinear` relies on tuned configurations (block sizes, num warps, etc.), which we provide for AF3 inputs to the kernel in `helper.py`. If you intend to apply the kernel to other input shapes, you can perform the Autotuning step (similar to `FusedEvoAttention`'s Step 3) with `untuned_fused_layernorm_linear.py`

</details>

---
### FusedTransition
The folder `FusedTransition` includes source code of FusedTransition kernel.

<details>
<summary>Expand for step-by-step guide</summary>

#### Step 1: Import

```
from fused_transition import FusedTransition
```

#### Step 2: In-code usage

`FusedTransition` fuses the AF3's Transition layer (original implementation in `reference/torch_transition.py`). You can replace the original Transition with `FusedTransition`.

```diff
# init
- transition = Transition(dim=dim, expansion_factor=expansion_factor)
+ transition = FusedTransition(dim=dim, expansion_factor=expansion_factor)
```

- **NOTE**: `FusedTransition` relies on FusedLayernormLinear for its expanding projections. Make sure you read FusedLayernormLinear's usage guide above. 

</details>


## Citation 

```
TODO: add bib
```

