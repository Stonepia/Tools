

# Original file: ./DebertaV2ForQuestionAnswering__0_forward_205.0/DebertaV2ForQuestionAnswering__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/xa/cxalb6k2gbrxn6ipcbqfcg6fmtw2aeq6expwodx5atzw647p5ozl.py
# Source Nodes: [trampoline_autograd_apply, trampoline_autograd_apply_1, trampoline_autograd_apply_2], Original ATen: [aten._softmax, aten._to_copy, aten.bernoulli, aten.bitwise_not, aten.lift_fresh, aten.masked_fill, aten.mul, aten.rsub]
# trampoline_autograd_apply => full_default_1
# trampoline_autograd_apply_1 => amax, div_1, exp, full_default_3, full_default_4, sub_2, sum_1, where_1, where_2
# trampoline_autograd_apply_2 => convert_element_type_3, convert_element_type_4, lt_1, mul_6, sub_3, where_3
triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_lift_fresh_masked_fill_mul_rsub_3 = async_compile.triton('triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_lift_fresh_masked_fill_mul_rsub_3', '''
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
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_lift_fresh_masked_fill_mul_rsub_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_bernoulli_bitwise_not_lift_fresh_masked_fill_mul_rsub_3(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 12288
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.full([1], False, tl.int1)
    tmp2 = -3.4028234663852886e+38
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.load(in_ptr1 + load_seed_offset)
    tmp15 = r1 + (512*x0)
    tmp16 = tl.rand(tmp14, (tmp15).to(tl.uint32))
    tmp17 = 0.9
    tmp18 = tmp16 < tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp19
    tmp22 = (tmp21 != 0)
    tmp23 = tmp9 / tmp13
    tmp24 = 0.0
    tmp25 = tl.where(tmp1, tmp24, tmp23)
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp27 = 1.1111111111111112
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp22, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp25, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp28, rmask)
''')
