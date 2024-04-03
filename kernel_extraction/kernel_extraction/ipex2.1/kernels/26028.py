

# Original file: ./hf_T5___60.0/hf_T5___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/m7/cm7m5nxdbvygbn27dy6xcmgmw7tgj4thv6arjc632nk4xu7ggpip.py
# Source Nodes: [add_4, add_5, clamp, l__mod___model_encoder_block_0_layer_0_dropout, l__mod___model_encoder_embed_tokens, mean_1, mul_5, mul_6, neg, pow_2, rsqrt_1, to_5, to_6, where_1], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.clone, aten.embedding, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_4 => add_6
# add_5 => add_7
# clamp => clamp_max, clamp_min, convert_element_type_8, convert_element_type_9
# l__mod___model_encoder_block_0_layer_0_dropout => clone_3
# l__mod___model_encoder_embed_tokens => embedding
# mean_1 => mean_1
# mul_5 => mul_5
# mul_6 => mul_6
# neg => neg
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# to_5 => convert_element_type_10
# to_6 => convert_element_type_11
# where_1 => full_default_2, full_default_3, where_1
triton_per_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_8 = async_compile.triton('triton_per_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_8', '''
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
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp21 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = 64504.0
    tmp9 = 65504.0
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = -tmp10
    tmp12 = triton_helpers.maximum(tmp5, tmp11)
    tmp13 = triton_helpers.minimum(tmp12, tmp10)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp15 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp21 * tmp28
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp29, rmask)
''')
