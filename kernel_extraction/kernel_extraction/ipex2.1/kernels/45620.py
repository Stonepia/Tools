

# Original file: ./T5ForConditionalGeneration__0_forward_169.0/T5ForConditionalGeneration__0_forward_169.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/yb/cybowvkdmxfdbr2ov65zpj3obhvklndc5pz7fom5qphh3pwq4ynz.py
# Source Nodes: [add_26, add_27, l__mod___encoder_block_5_layer__1__dropout, l__mod___encoder_dropout_1, mean_12, mul_27, mul_28, pow_13, rsqrt_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt]
# add_26 => add_33
# add_27 => add_34
# l__mod___encoder_block_5_layer__1__dropout => gt_25, mul_75, mul_76
# l__mod___encoder_dropout_1 => gt_26, mul_79, mul_80
# mean_12 => mean_12
# mul_27 => mul_77
# mul_28 => mul_78
# pow_13 => pow_13
# rsqrt_12 => rsqrt_12
triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_9 = async_compile.triton('triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_9', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_9(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, load_seed_offset1, xnumel, rnumel):
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tl.load(in_ptr0 + load_seed_offset1)
    tmp23 = tl.rand(tmp22, (tmp1).to(tl.uint32))
    tmp24 = tmp23 > tmp3
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp11 * tmp21
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 * tmp28
    tmp30 = tmp29 * tmp9
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp4, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp11, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp21, None)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp24, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp30, rmask)
''')
