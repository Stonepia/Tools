

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/o7/co7iuciw2rwkyjzsi3e72pngb46qpqzb7lkxoyogdtj3rhwe44t2.py
# Source Nodes: [cross_entropy, to_69], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum, aten.where]
# cross_entropy => full_default_67
# to_69 => convert_element_type_167
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_5 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_5', '''
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp32', 6: '*i1', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask)
    tmp1 = 0.04419417382415922
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp20 = tmp10 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = -0.5
    tmp23 = tmp17 * tmp22
    tmp24 = tmp19 * tmp19
    tmp25 = tmp24 * tmp19
    tmp26 = tmp23 * tmp25
    tmp27 = 512.0
    tmp28 = tmp26 / tmp27
    tmp29 = 2.0
    tmp30 = tmp12 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp21 + tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp18, tmp33, tmp34)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 * tmp5
    tmp39 = tmp35 * tmp38
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp35, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp39, rmask)
''')
