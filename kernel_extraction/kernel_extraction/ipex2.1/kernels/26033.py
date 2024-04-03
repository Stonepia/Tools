

# Original file: ./hf_T5___60.0/hf_T5___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/uj/cujncxiisco63kjd5yqdjdgyr42zasoik66t6oejt5utb2ihqocz.py
# Source Nodes: [add_8, add_9, clamp_1, clamp_2, l__mod___model_encoder_block_1_layer_0_dropout, mean_3, mul_10, mul_9, neg_1, neg_2, pow_4, rsqrt_3, to_10, to_9, where_1, where_2, where_3], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.clone, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_8 => add_11
# add_9 => add_12
# clamp_1 => clamp_max_1, clamp_min_1, convert_element_type_12, convert_element_type_13
# clamp_2 => clamp_max_2, clamp_min_2, convert_element_type_18, convert_element_type_19
# l__mod___model_encoder_block_1_layer_0_dropout => clone_8
# mean_3 => mean_3
# mul_10 => mul_10
# mul_9 => mul_9
# neg_1 => neg_1
# neg_2 => neg_2
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
# to_10 => convert_element_type_21
# to_9 => convert_element_type_20
# where_1 => full_default_2, full_default_3
# where_2 => where_2
# where_3 => where_3
triton_per_fused__to_copy_add_clamp_clone_mean_mul_neg_pow_rsqrt_scalar_tensor_where_13 = async_compile.triton('triton_per_fused__to_copy_add_clamp_clone_mean_mul_neg_pow_rsqrt_scalar_tensor_where_13', '''
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_clone_mean_mul_neg_pow_rsqrt_scalar_tensor_where_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_clone_mean_mul_neg_pow_rsqrt_scalar_tensor_where_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp27 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = 64504.0
    tmp5 = 65504.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = -tmp6
    tmp8 = triton_helpers.maximum(tmp1, tmp7)
    tmp9 = triton_helpers.minimum(tmp8, tmp6)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tl.where(tmp15, tmp4, tmp5)
    tmp17 = -tmp16
    tmp18 = triton_helpers.maximum(tmp13, tmp17)
    tmp19 = triton_helpers.minimum(tmp18, tmp16)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = 512.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp21 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp27 * tmp34
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp35, rmask)
''')
