

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/ka/cka6s66hheh6v6bsx26ws7ma4gy42mnadgdosj3oksfp3vpp5qwm.py
# Source Nodes: [add_6, add_7, clamp_1, l__mod___encoder_block_0_layer__1__dropout, l__mod___encoder_block_1_layer_0_self_attention_q, mean_2, mul_7, mul_8, neg_1, pow_3, rsqrt_2, to_7, to_8, where_1, where_2], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.mean, aten.mul, aten.native_dropout, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.view, aten.where]
# add_6 => add_8
# add_7 => add_9
# clamp_1 => clamp_max_1, clamp_min_1, convert_element_type_12, convert_element_type_13
# l__mod___encoder_block_0_layer__1__dropout => mul_15, mul_16
# l__mod___encoder_block_1_layer_0_self_attention_q => view_26
# mean_2 => mean_2
# mul_7 => mul_17
# mul_8 => mul_18
# neg_1 => neg_1
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
# to_7 => convert_element_type_14
# to_8 => convert_element_type_15
# where_1 => full_default_2, full_default_3
# where_2 => where_2
triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_11 = async_compile.triton('triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_11', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*i1', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_mean_mul_native_dropout_neg_pow_rsqrt_scalar_tensor_view_where_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp29 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp30 = tmp18 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp17, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp28, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp32, rmask)
''')
