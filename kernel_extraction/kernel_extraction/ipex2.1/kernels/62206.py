

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/lr/clrcvfjotm7vtynj73p356ot5cir37fsqkxpkumssvreycqthnmf.py
# Source Nodes: [add_4, add_5, clamp, l__mod___encoder_block_0_layer_0_dropout, l__mod___encoder_block_0_layer__1__dense_relu_dense_wi, l__mod___encoder_dropout, mean_1, mul_5, mul_6, neg, pow_2, rsqrt_1, to_5, to_6, where_1], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.mean, aten.mul, aten.native_dropout, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.view, aten.where]
# add_4 => add_6
# add_5 => add_7
# clamp => clamp_max, clamp_min, convert_element_type_8, convert_element_type_9
# l__mod___encoder_block_0_layer_0_dropout => mul_10, mul_9
# l__mod___encoder_block_0_layer__1__dense_relu_dense_wi => view_22
# l__mod___encoder_dropout => mul_1, mul_2
# mean_1 => mean_1
# mul_5 => mul_11
# mul_6 => mul_12
# neg => neg
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# to_5 => convert_element_type_10
# to_6 => convert_element_type_11
# where_1 => full_default_2, full_default_3, where_1
triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_8 = async_compile.triton('triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_8', '''
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: '*i1', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask)
    tmp2 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask)
    tmp8 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (0))
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp4
    tmp11 = tmp5 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp15 = 64504.0
    tmp16 = 65504.0
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = -tmp17
    tmp19 = triton_helpers.maximum(tmp12, tmp18)
    tmp20 = triton_helpers.minimum(tmp19, tmp17)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = 512.0
    tmp29 = tmp27 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp34 = tmp22 * tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 * tmp35
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp21, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp36, rmask)
''')
