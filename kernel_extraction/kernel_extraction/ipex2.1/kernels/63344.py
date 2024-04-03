

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/rj/crjdcftd35jxy63fjzu73yo465knutdv4udj7cqwjhzpml7feuq4.py
# Source Nodes: [to_89], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
# to_89 => convert_element_type_139
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_5 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_5', '''
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
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*i1', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp8 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = -0.5
    tmp20 = tmp15 * tmp19
    tmp21 = tmp16 * tmp16
    tmp22 = tmp21 * tmp16
    tmp23 = tmp20 * tmp22
    tmp24 = 512.0
    tmp25 = tmp23 / tmp24
    tmp26 = 2.0
    tmp27 = tmp10 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp18 + tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp3
    tmp34 = tmp30 * tmp33
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp30, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp34, rmask)
''')
