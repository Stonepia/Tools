

# Original file: ./T5ForConditionalGeneration__0_forward_169.0/T5ForConditionalGeneration__0_forward_169.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/xt/cxt5j5nhykzq6nzz3iakzigsqh6l6iu6cgwkkc4jhbar7pqycubu.py
# Source Nodes: [dropout_6, float_9, softmax_6, type_as_6], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# dropout_6 => gt_28, mul_89, mul_90
# float_9 => convert_element_type_49
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
# type_as_6 => convert_element_type_50
triton_per_fused__softmax__to_copy_native_dropout_11 = async_compile.triton('triton_per_fused__softmax__to_copy_native_dropout_11', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_native_dropout_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_native_dropout_11(in_ptr0, in_ptr1, in_ptr2, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 8
    tmp0 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, other=0.0).to(tl.float32)
    tmp1 = (-1)*(tl.minimum(0, r3 + ((-1)*x0), tl.PropagateNan.NONE))
    tmp2 = tl.full([1], 16, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tmp1.to(tl.float32)
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.log(tmp6)
    tmp8 = 2.0794415416798357
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9 * tmp5
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tmp11 + tmp2
    tmp13 = tl.full([1], 31, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tl.where(tmp3, tmp1, tmp14)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.where(tmp17 < 0, tmp17 + 32, tmp17)
    # tl.device_assert((0 <= tmp18) & (tmp18 < 32), "index out of bounds: 0 <= tmp18 < 32")
    tmp19 = tl.load(in_ptr1 + (x1 + (8*tmp18)), None).to(tl.float32)
    tmp20 = r3
    tmp21 = x0
    tmp22 = tmp20 <= tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = -3.3895313892515355e+38
    tmp28 = tmp26 * tmp27
    tmp29 = tmp19 + tmp28
    tmp30 = tmp0 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask, tmp32, float("-inf"))
    tmp35 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp34, 0))
    tmp36 = tmp31 - tmp35
    tmp37 = tl.exp(tmp36)
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp42 = tl.load(in_ptr2 + load_seed_offset)
    tmp43 = r3 + (1024*x4)
    tmp44 = tl.rand(tmp42, (tmp43).to(tl.uint32))
    tmp45 = tmp44.to(tl.float32)
    tmp46 = 0.1
    tmp47 = tmp45 > tmp46
    tmp48 = tmp37 / tmp41
    tmp49 = tmp47.to(tl.float32)
    tmp50 = tmp48.to(tl.float32)
    tmp51 = tmp49 * tmp50
    tmp52 = 1.1111111111111112
    tmp53 = tmp51 * tmp52
    tl.store(out_ptr4 + (r3 + (1024*x4)), tmp47, rmask)
    tl.store(out_ptr5 + (r3 + (1024*x4)), tmp48, rmask)
    tl.store(out_ptr6 + (r3 + (1024*x4)), tmp53, rmask)
''')
