

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/v2/cv2dsmz5ybjb54z2cfhbgc3kan3hroofofk6u6j4mssjvmlmyxhs.py
# Source Nodes: [to_67], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
# to_67 => convert_element_type_105
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*i1', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp3 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 + tmp14
    tmp16 = -0.5
    tmp17 = tmp10 * tmp16
    tmp18 = tmp12 * tmp12
    tmp19 = tmp18 * tmp12
    tmp20 = tmp17 * tmp19
    tmp21 = 512.0
    tmp22 = tmp20 / tmp21
    tmp23 = 2.0
    tmp24 = tmp5 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp15 + tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 1.1111111111111112
    tmp31 = tmp29 * tmp30
    tmp32 = tmp27 * tmp31
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp27, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp32, rmask)
''')
