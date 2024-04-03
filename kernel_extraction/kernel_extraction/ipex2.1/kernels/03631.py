

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/mp/cmpikex64uxfyrlrsx2cdwl2jhzbslslduanq6tepxrz7n6dmzmy.py
# Source Nodes: [l__mod___decoder_dropout, to_32], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# l__mod___decoder_dropout => mul_84, mul_85
# to_32 => convert_element_type_69
triton_per_fused__to_copy_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_pow_sum_27 = async_compile.triton('triton_per_fused__to_copy_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_pow_sum_27', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*i1', 6: '*fp16', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_pow_sum_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_pow_sum_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
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
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp10 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp22 = tmp7 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 + tmp23
    tmp25 = -0.5
    tmp26 = tmp19 * tmp25
    tmp27 = tmp21 * tmp21
    tmp28 = tmp27 * tmp21
    tmp29 = tmp26 * tmp28
    tmp30 = 512.0
    tmp31 = tmp29 / tmp30
    tmp32 = 2.0
    tmp33 = tmp14 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp24 + tmp35
    tmp38 = tl.where(tmp37 < 0, tmp37 + 32128, tmp37)
    tmp39 = tl.full([1], -1, tl.int64)
    tmp40 = tmp37 == tmp39
    tmp41 = tmp9 * tmp12
    tmp42 = tmp36 * tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 0.0
    tmp45 = tl.where(tmp40, tmp44, tmp43)
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(r1 + (512*tmp38), [RBLOCK])), tmp45, rmask)
''')
