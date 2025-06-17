# MegaFold: System-Level Optimizations for Accelerating Protein Structure Prediction Models

## About 

[MegaFold](TODO:add arxiv link) is a cross-platform system to accelerate protein structure prediction models (e.g., AlphaFold3, AlphaFold2).

Why MegaFold? 

- **Cross-platform support**: Supports execution on heterogeneous devices, including NVIDIA GPUs, AMD GPUs, and CPUs.
- **Ease of use**: Delivers huge performance gains with few lines of code change
- **Speed improvement**: Accelerates per-iteration training time by up to 1.73x
- **Memory reduction**: Reduces peak memory during training by up to 1.23x
- **Sequence length extension**: Enables training on 1.35x longer sequence lengths


## Usage

We include code for AlphaFold3 training (both baseline and end-to-end MegaFold integrations) and instructions to reproduce our paper results. 

<details>
<summary>Expand for running AF3 guide</summary>

### Install required dependencies

```
# create virtual environment under python==3.13 and activate 
conda create -n venv python==3.13.0
conda activate venv 

# install torch==2.7.0+cu11.8
pip install torch==2.7.0  --index-url https://download.pytorch.org/whl/cu118

# install other packages
pip install -r requirements.txt
```

---
### Prepare experiment dataset

First, download a sample dataset from the Protein Data Bank (PDB). 

```
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/ESbEXPguyO9Moh3E_J1zkWQBXZ6JxE5bsoKrZXOVwtu1Ow?download=1" -O data/pdb_data/val_mmcifs.tar.gz
tar -xzf data/pdb_data/val_mmcifs.tar.gz -C data/pdb_data
rm data/pdb_data/val_mmcifs.tar.gz
```

Then, install required MSAs and templates data.

```
# install msa_dir
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/EbXU1bnlRZxIqUXbAprgHycB3F4GWLy-m-qxvODfJsvFvA?download=1" -O pdb_val_msas
tar -xvzf pdb_val_msas
cp -r scratch/references/af3/pdb_data/* data/pdb_data/
rm pdb_val_msas
rm -r scratch

# install templates_dir
wget "https://umass-my.sharepoint.com/:u:/g/personal/hvla_umass_edu/EUalS7Hq3KBOlGdF2bVVwFABYU_ZidT2nEEi0PwqxaZ_Fw?download=1" -O templates_dir
tar -xvzf templates_dir 
cp -r scratch/references/af3/pdb_data/* data/pdb_data/
rm templates_dir
rm -r scratch
```

Then, install PDB's Chemical Component Dictionary (CCD) and miscellaneous metadata. 

```
# install CCD data
wget -P ./data/ccd_data/ https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
wget -P ./data/ccd_data/ https://files.wwpdb.org/pub/pdb/data/component-models/complete/chem_comp_model.cif.gz
gunzip data/ccd_data/components.cif.gz
gunzip data/ccd_data/chem_comp_model.cif.gz

# install misc_data
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/ESb9kUT_ASBEsYRN0KQmqt4BLzJhFunQU86E-GxWGxtGiA?download=1" -O misc_data
tar -xzf misc_data -C data/pdb_data
rm misc_data
```

Now, download the cache of deterministic features, used in Ahead-of-Time Cache-based Data-Loading Optimization.

```
# install msa_cache_dir
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/Ect3VyxyqnZPm-4I6EpzB64B2M6tGctY5OMjIkatr6kYHQ?download=1" -O msa_cache
tar -xvzf msa_cache --wildcards 'caches/pdb_data/cache/msa/val_msas*'
rm msa_cache

# install input_cache_dir 
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/EXQnFYxhepNNku_Df45B1gEBPlhzIH_RtnhUEae4b74SKQ?download=1" -O input_cache
tar -xvzf input_cache 
rm input_cache
```

---
### Run code

``` 
AF3_OPTIMIZATIONS_MODE="megafold" python3 train.py --config configs/megafold_interactive.yaml --trainer_name initial_training
```

Script to submit batch jobs is available in `scripts`. For example, you want to launch a job with `nodes=1` and `gpus=2`: 

```
sbatch --nodes=1 --ntasks-per-node=2 --gpus=2 scripts/megafold.sh
```


---
### (optional) Full dataset & cache:

If you are interested in running large-scale AlphaFold3 training, the full dataset and its cache are provided below:  

```
# download `omniflow_caches.tar.gz.part_{aa,ab}` and `omniflow_data.tar.gz` from SharePoint
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/Ect3VyxyqnZPm-4I6EpzB64B2M6tGctY5OMjIkatr6kYHQ?download=1"
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/ERiOg_fC_6BFnr9oKilzeeUBz8O_a2tI0i-TlksYAf8E5g?download=1"
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/EYQ9oFu5KmFLryp8F1m79BAB2zoUFtLIU-Bx2OWmmKAdtA?download=1"

# then reassemble, extract, and clean up the downloaded archives
cat omniflow_caches.tar.gz.part_* > omniflow_caches.tar.gz
tar -xzf omniflow_caches.tar.gz && rm omniflow_caches.tar.gz
tar -xzf omniflow_data.tar.gz && rm omniflow_data.tar.gz
```
</details>


---

The following section gives detailed instructions on enabling each of our optimizations.


### Optimization 1: Data-loading
The file `megafold/inputs.py` includes the data pipeline and implementation details for the ahead-of-time cache-based data loading optimizations. 

You can find details on [deterministic input features cache](https://github.com/Supercomputing-System-AI-Lab/MegaFold/blob/main/megafold/inputs.py#L4536-L4553) and on [MSA features cache](https://github.com/Supercomputing-System-AI-Lab/MegaFold/blob/main/megafold/inputs.py#L4670-L4732).

---
### Optimization 2: FusedEvoAttention
The folder `FusedEvoAttention` includes source code of FusedEvoAttention kernel. 

<details>
<summary>Expand for step-by-step guide</summary>

#### Step 1: Import

```
from evoattention import TritonEvoformer
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
### Optimization 3: FusedLayernormLinear
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

- **NOTE**: `LayernormLinear` relies on tuned configurations (block sizes, num warps, etc.), which we provide for AF3 inputs to the kernel in `helper.py`. If you intend to apply the kernel to other input shapes, you can perform the Autotuning step (similar to `FusedEvoAttention`'s Step 3) with `untuned_fused_layernorm_linear.py`

</details>

---
### Optimization 4: FusedTransition
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

