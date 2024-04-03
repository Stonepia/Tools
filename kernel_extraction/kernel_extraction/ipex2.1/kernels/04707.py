

# Original file: ./DebertaForMaskedLM__0_forward_133.0/DebertaForMaskedLM__0_forward_133.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/cb/ccbudbk2tg3pjpyasacx775xgaiksrgczmvxhjz2gufffnzqtfqa.py
# Source Nodes: [trampoline_autograd_apply_1, trampoline_autograd_apply_2], Original ATen: [aten._softmax, aten._to_copy, aten.bernoulli, aten.bitwise_not, aten.masked_fill, aten.mul, aten.rsub]
# trampoline_autograd_apply_1 => amax, convert_element_type_7, convert_element_type_8, div_2, exp, full_default_3, full_default_4, full_default_5, sub_3, sum_1, where_1, where_2
# trampoline_autograd_apply_2 => convert_element_type_10, convert_element_type_9, lt_1, mul_5, sub_4, where_3
triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5 = async_compile.triton('triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5', '''
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*i1', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_masked_fill_mul_rsub_5(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 49152
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
    tmp1 = tl.full([1], False, tl.int1)
    tmp2 = -65504.0
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp7, 0))
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.load(in_ptr1 + load_seed_offset)
    tmp16 = r1 + (512*x0)
    tmp17 = tl.rand(tmp15, (tmp16).to(tl.uint32))
    tmp18 = 0.9
    tmp19 = tmp17 < tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1.0
    tmp22 = tmp21 - tmp20
    tmp23 = (tmp22 != 0)
    tmp24 = tmp10 / tmp14
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 0.0
    tmp27 = tl.where(tmp1, tmp26, tmp25)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = 1.1111111111111112
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp23, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp27, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp30, rmask)
''')
