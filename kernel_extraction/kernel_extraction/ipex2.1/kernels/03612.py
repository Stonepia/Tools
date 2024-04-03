

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/cc/cccuwamncbg7dxojnfcmcx77sk2ui22nypssa7dagoaxvrdui3e7.py
# Source Nodes: [cross_entropy, to_67], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum, aten.where]
# cross_entropy => full_default_67
# to_67 => convert_element_type_163
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp32', 6: '*i1', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
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
    tmp11 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask)
    tmp12 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp12 + tmp15
    tmp17 = -0.5
    tmp18 = tmp10 * tmp17
    tmp19 = tmp13 * tmp13
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 * tmp20
    tmp22 = 512.0
    tmp23 = tmp21 / tmp22
    tmp24 = 2.0
    tmp25 = tmp5 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp16 + tmp27
    tmp29 = 0.0
    tmp30 = tl.where(tmp11, tmp28, tmp29)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 1.1111111111111112
    tmp34 = tmp32 * tmp33
    tmp35 = tmp30 * tmp34
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp30, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp35, rmask)
''')
