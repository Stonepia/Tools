

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/6u/c6uhe4tosez22plqy3nbfrhyuo7n52erh2wgw65rvgerfeb7me7o.py
# Source Nodes: [add_4, add_5, l__mod___encoder_block_0_layer_0_dropout, l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0, l__mod___encoder_dropout, mean_1, mul_5, mul_6, pow_2, rsqrt_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt, aten.view]
# add_4 => add_6
# add_5 => add_7
# l__mod___encoder_block_0_layer_0_dropout => gt_3, mul_10, mul_9
# l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0 => view_22
# l__mod___encoder_dropout => mul_1, mul_2
# mean_1 => mean_1
# mul_5 => mul_11
# mul_6 => mul_12
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_view_6 = async_compile.triton('triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_view_6', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_view_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_mean_mul_native_dropout_pow_rsqrt_view_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, load_seed_offset, xnumel, rnumel):
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
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp26 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 * tmp9
    tmp15 = tmp10 + tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 512.0
    tmp22 = tmp20 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp27 = tmp15 * tmp25
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp4, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp15, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp25, None)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp28, rmask)
''')
