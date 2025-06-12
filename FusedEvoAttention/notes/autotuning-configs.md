# H200

## Seq len = 64: 
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=16, num_stages=3, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4,

## Seq len = 128:
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=32, num_stages=2, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4,

## Seq len = 192: 
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=32, num_stages=3, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4,

## Seq len = 256:
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=16, num_stages=2, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=3, num_warps=4,

## Seq len = 384
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=3, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=32, num_stages=2, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_stages=3, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=3, num_warps=4,

## Seq len = 512:
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=3, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=32, num_stages=2, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=32, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=3, num_warps=4,

## Seq len = 640:
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=64, num_stages=3, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=16, num_stages=2, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64, num_stages=3, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=3, num_warps=4,



# MI250

## seq len = 64: 

_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=4, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=16, num_stages=3, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=1, num_warps=8,

## seq len = 96: 

_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=3, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=16, num_stages=3, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=1, num_warps=4,

## Seq len = 128:
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=4, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=16, num_stages=3, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=1, num_warps=4,

## Seq len = 256:
_attn_fwd: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=1, num_warps=4
_attn_bwd_preprocess: BLOCK_SIZE_Q=32, num_stages=3, num_warps=4
_attn_bwd_dk_dv: BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4, 
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=32, num_stages=1, num_warps=4,

## Seq len = 384
_attn_fwd: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4,
_attn_bwd_preprocess: BLOCK_SIZE_Q=16, num_stages=2, num_warps=4,
_attn_bwd_dk_dv: BLOCK_SIZE_Q=64, BLOCK_SIZE_KV=64, num_stages=1, num_warps=4,
_attn_bwd_dq: BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=16, num_stages=1, num_warps=4
