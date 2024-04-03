

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/ya/cyagw7n6l543qpnl3d3ebgm3olwbvfhjkq5wlgdabvqd5mcfamhw.py
# Source Nodes: [add_50, add_51, clamp_15, l__mod___encoder_block_7_layer__1__dropout, l__mod___encoder_dropout_1, mean_16, mul_75, mul_76, neg_15, pow_25, rsqrt_16, to_35, to_36, where_1, where_16], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.mean, aten.mul, aten.native_dropout, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_50 => add_59
# add_51 => add_60
# clamp_15 => clamp_max_15, clamp_min_15, convert_element_type_82, convert_element_type_83
# l__mod___encoder_block_7_layer__1__dropout => mul_139, mul_140
# l__mod___encoder_dropout_1 => gt_34, mul_143, mul_144
# mean_16 => mean_16
# mul_75 => mul_141
# mul_76 => mul_142
# neg_15 => neg_15
# pow_25 => pow_25
# rsqrt_16 => rsqrt_16
# to_35 => convert_element_type_84
# to_36 => convert_element_type_85
# where_1 => full_default_2, full_default_3
# where_16 => where_16
triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_where_12 = async_compile.triton('triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_where_12', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*i1', 5: '*i64', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: '*fp16', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_where_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_where_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp11 = 64504.0
    tmp12 = 65504.0
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = -tmp13
    tmp15 = triton_helpers.maximum(tmp8, tmp14)
    tmp16 = triton_helpers.minimum(tmp15, tmp13)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = 512.0
    tmp25 = tmp23 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tl.load(in_ptr4 + load_seed_offset)
    tmp30 = r1 + (512*x0)
    tmp31 = tl.rand(tmp29, (tmp30).to(tl.uint32))
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 0.1
    tmp34 = tmp32 > tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp37 = tmp18 * tmp28
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 * tmp38
    tmp40 = tmp35 * tmp39
    tmp41 = tmp40 * tmp5
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp17, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp28, None)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp34, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp41, rmask)
''')
