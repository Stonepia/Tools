

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/xr/cxr7rhwowdayjyyux7gnhlpnhwwdikpnj6dstiloul73lvd6pjq5.py
# Source Nodes: [add_55, add_57, add_59], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
# add_55 => add_70
# add_57 => add_73
# add_59 => add_76
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_14 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_14', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*i1', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp21 = tmp3 * tmp20
    tmp22 = tmp19 + tmp21
    tmp23 = -0.5
    tmp24 = tmp18 * tmp23
    tmp25 = tmp20 * tmp20
    tmp26 = tmp25 * tmp20
    tmp27 = tmp24 * tmp26
    tmp28 = 512.0
    tmp29 = tmp27 / tmp28
    tmp30 = 2.0
    tmp31 = tmp13 * tmp30
    tmp32 = tmp29 * tmp31
    tmp33 = tmp22 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = 1.1111111111111112
    tmp38 = tmp36 * tmp37
    tmp39 = tmp34 * tmp38
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp33, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp39, rmask)
''')
