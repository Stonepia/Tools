

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/di/cdi5zq3sog7q2viqyuyrofttqjdenv5v7e4gymnxeqcf46zm5oyb.py
# Source Nodes: [add_119, add_120, clamp_39, l__mod___decoder_block_7_layer__1__dropout, l__mod___decoder_dropout_1, l__mod___lm_head, mean_41, mul_169, mul_170, neg_40, pow_58, rsqrt_41, to_89, to_90, where_1, where_41], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.ge, aten.le, aten.logical_and, aten.mean, aten.mul, aten.native_dropout, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.view, aten.where]
# add_119 => add_144
# add_120 => add_145
# clamp_39 => clamp_max_39, clamp_min_39, convert_element_type_217, convert_element_type_218
# l__mod___decoder_block_7_layer__1__dropout => mul_333, mul_334
# l__mod___decoder_dropout_1 => gt_84, mul_337, mul_338
# l__mod___lm_head => view_602
# mean_41 => mean_41
# mul_169 => mul_335
# mul_170 => mul_336
# neg_40 => neg_40
# pow_58 => pow_58
# rsqrt_41 => rsqrt_41
# to_89 => convert_element_type_219
# to_90 => convert_element_type_220
# where_1 => full_default_2, full_default_3
# where_41 => where_41
triton_per_fused__to_copy_add_clamp_ge_le_logical_and_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_17 = async_compile.triton('triton_per_fused__to_copy_add_clamp_ge_le_logical_and_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_17', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*i1', 5: '*i64', 6: '*fp16', 7: '*fp16', 8: '*i1', 9: '*i1', 10: '*fp16', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_ge_le_logical_and_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_ge_le_logical_and_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
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
    tmp41 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp18 = tmp14.to(tl.float32)
    tmp19 = tmp7 >= tmp18
    tmp20 = tmp13.to(tl.float32)
    tmp21 = tmp7 <= tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp17.to(tl.float32)
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = 512.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-06
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tl.load(in_ptr4 + load_seed_offset)
    tmp35 = r1 + (512*x0)
    tmp36 = tl.rand(tmp34, (tmp35).to(tl.uint32))
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 0.1
    tmp39 = tmp37 > tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp42 = tmp23 * tmp33
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 * tmp43
    tmp45 = tmp40 * tmp44
    tmp46 = tmp45 * tmp5
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp17, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp33, None)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp39, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp46, rmask)
''')
