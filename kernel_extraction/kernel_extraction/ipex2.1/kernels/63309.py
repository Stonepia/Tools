

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/pt/cptl6u5yvebjob6ltlisqekmegywf7dvfjmyxg73yju6dgcusmdt.py
# Source Nodes: [cross_entropy, to_89], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum, aten.where]
# cross_entropy => full_default_87
# to_89 => convert_element_type_219
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp32', 6: '*i1', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp16 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask)
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
    tmp18 = tmp8 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = -0.5
    tmp21 = tmp15 * tmp20
    tmp22 = tmp17 * tmp17
    tmp23 = tmp22 * tmp17
    tmp24 = tmp21 * tmp23
    tmp25 = 512.0
    tmp26 = tmp24 / tmp25
    tmp27 = 2.0
    tmp28 = tmp10 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp19 + tmp30
    tmp32 = 0.0
    tmp33 = tl.where(tmp16, tmp31, tmp32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp3
    tmp37 = tmp33 * tmp36
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp33, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp37, rmask)
''')
