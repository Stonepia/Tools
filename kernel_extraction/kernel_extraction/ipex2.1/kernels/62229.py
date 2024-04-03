

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ia/ciaatf2uzhriedugny6qad7fdkznqfoyh6fsm3jalhhbtsmk57q7.py
# Source Nodes: [add_4, add_5, l__self___encoder_block_0_layer_0_dropout, l__self___encoder_block_0_layer__1__dense_relu_dense_wi, l__self___encoder_dropout, mean_1, mul_5, mul_6, pow_2, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt, aten.view]
# add_4 => add_6
# add_5 => add_7
# l__self___encoder_block_0_layer_0_dropout => gt_3, mul_10, mul_9
# l__self___encoder_block_0_layer__1__dense_relu_dense_wi => convert_element_type_13, view_22
# l__self___encoder_dropout => mul_1, mul_2
# mean_1 => mean_1
# mul_5 => mul_11
# mul_6 => mul_12
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_7 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_7', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, load_seed_offset, xnumel, rnumel):
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
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp16 = tmp10.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = 512.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp29 = tmp17 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp10, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask)
''')
