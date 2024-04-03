

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/j3/cj3av6325m47oqpzkp43oyxa3xqdx374jhehma52citxvktrhetv.py
# Source Nodes: [add_4, add_6, add_8, add_9, l__self___encoder_block_1_layer_0_dropout, l__self___encoder_block_1_layer__1__dense_relu_dense_wi, l__self___encoder_dropout, mean_3, mul_10, mul_9, pow_4, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt, aten.view]
# add_4 => add_6
# add_6 => add_8
# add_8 => add_11
# add_9 => add_12
# l__self___encoder_block_1_layer_0_dropout => gt_7, mul_21, mul_22
# l__self___encoder_block_1_layer__1__dense_relu_dense_wi => convert_element_type_28, view_47
# l__self___encoder_dropout => mul_1, mul_2
# mean_3 => mean_3
# mul_10 => mul_24
# mul_9 => mul_23
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_11 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_11', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp16', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp13 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 * tmp9
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp10.to(tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = 512.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-06
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp35 = tmp23 * tmp33
    tmp36 = tmp34 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp10, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp23, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp33, None)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp37, rmask)
''')
