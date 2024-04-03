

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/3v/c3vrycgitt3p4yhnzchhe5eoekcmorrdadvdbxfpgkjkdhkzt3g5.py
# Source Nodes: [cross_entropy, to_87], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum, aten.where]
# cross_entropy => full_default_87
# to_87 => convert_element_type_215
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_8 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_8', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*i1', 6: '*fp32', 7: '*i1', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp14 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp16 = tmp5 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp14 + tmp17
    tmp19 = -0.5
    tmp20 = tmp12 * tmp19
    tmp21 = tmp15 * tmp15
    tmp22 = tmp21 * tmp15
    tmp23 = tmp20 * tmp22
    tmp24 = 512.0
    tmp25 = tmp23 / tmp24
    tmp26 = 2.0
    tmp27 = tmp7 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp18 + tmp29
    tmp31 = 0.0
    tmp32 = tl.where(tmp13, tmp30, tmp31)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 1.1111111111111112
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 * tmp36
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp32, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp37, rmask)
''')
