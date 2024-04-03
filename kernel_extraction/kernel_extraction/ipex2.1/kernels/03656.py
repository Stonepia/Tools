

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/ml/cmlo5tlojbkab7olf4utaog5o7jps3t6iedwn234cilkohiblhuy.py
# Source Nodes: [add_32, add_35, l__self___decoder_dropout], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# add_32 => add_40
# add_35 => add_44
# l__self___decoder_dropout => mul_84, mul_85
triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_21 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_21', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: '*fp32', 8: '*i1', 9: '*bf16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
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
    tmp4 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask)
    tmp6 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp3 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp3 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = -0.5
    tmp26 = tmp20 * tmp25
    tmp27 = tmp22 * tmp22
    tmp28 = tmp27 * tmp22
    tmp29 = tmp26 * tmp28
    tmp30 = 512.0
    tmp31 = tmp29 / tmp30
    tmp32 = 2.0
    tmp33 = tmp15 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp24 + tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38 * tmp8
    tmp40 = tmp36 * tmp39
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp35, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp40, rmask)
''')
