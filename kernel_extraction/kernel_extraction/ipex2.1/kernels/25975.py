

# Original file: ./hf_T5___60.0/hf_T5___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ys/cys7lvcze3mkweibqfoesvgz5qzk72fgubkcn4xendv6tja4l2cn.py
# Source Nodes: [add_4, add_6, add_8, add_9, l__mod___model_encoder_block_0_layer_0_dropout, l__mod___model_encoder_block_0_layer__1__dropout, l__mod___model_encoder_block_1_layer_0_dropout, l__mod___model_encoder_embed_tokens, mean_3, mul_10, mul_9, pow_4, rsqrt_3, to_10, to_9], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_4 => add_6
# add_6 => add_8
# add_8 => add_11
# add_9 => add_12
# l__mod___model_encoder_block_0_layer_0_dropout => clone_3
# l__mod___model_encoder_block_0_layer__1__dropout => clone_5
# l__mod___model_encoder_block_1_layer_0_dropout => clone_8
# l__mod___model_encoder_embed_tokens => embedding
# mean_3 => mean_3
# mul_10 => mul_10
# mul_9 => mul_9
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
# to_10 => convert_element_type_15
# to_9 => convert_element_type_14
triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp15 * tmp22
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp23, rmask)
''')
