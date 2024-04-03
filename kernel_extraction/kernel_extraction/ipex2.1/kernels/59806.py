

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/rt/crtdy5fehxqjkrelnxsu6ftxqeuurdfjee5uri344o4rijmhd33p.py
# Source Nodes: [to_27], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]
# to_27 => convert_element_type_40
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_19 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_19', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*i1', 14: '*bf16', 15: '*bf16', 16: '*fp32', 17: '*i1', 18: '*bf16', 19: '*bf16', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_pow_sum_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask)
    tmp30 = tl.load(in_ptr13 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tmp29 = tmp24 * tmp28
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp41 = tmp32 * tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = -0.5
    tmp44 = tmp39 * tmp43
    tmp45 = tmp40 * tmp40
    tmp46 = tmp45 * tmp40
    tmp47 = tmp44 * tmp46
    tmp48 = 512.0
    tmp49 = tmp47 / tmp48
    tmp50 = 2.0
    tmp51 = tmp34 * tmp50
    tmp52 = tmp49 * tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp42 + tmp53
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 * tmp27
    tmp58 = tmp54 * tmp57
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp29, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp54, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp58, rmask)
''')
