

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/dg/cdg4gbls2majj5qnk6ahj7sqig7qotyl6kchypeyuhvvpc3qpvho.py
# Source Nodes: [add_4, add_6, add_8, add_9, l__self___model_encoder_block_0_layer_0_dropout, l__self___model_encoder_block_0_layer__1__dropout, l__self___model_encoder_block_1_layer_0_dropout, l__self___model_encoder_block_1_layer__1__dense_relu_dense_wi, l__self___model_encoder_embed_tokens, mean_3, mul_10, mul_9, pow_4, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_4 => add_6
# add_6 => add_8
# add_8 => add_11
# add_9 => add_12
# l__self___model_encoder_block_0_layer_0_dropout => clone_3
# l__self___model_encoder_block_0_layer__1__dropout => clone_5
# l__self___model_encoder_block_1_layer_0_dropout => clone_8
# l__self___model_encoder_block_1_layer__1__dense_relu_dense_wi => convert_element_type_28
# l__self___model_encoder_embed_tokens => embedding
# mean_3 => mean_3
# mul_10 => mul_10
# mul_9 => mul_9
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (r1 + (1024*tmp1)), rmask & xmask, other=0.0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 1024.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp11 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp11, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')
