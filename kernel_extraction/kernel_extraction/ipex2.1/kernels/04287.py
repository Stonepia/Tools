

# Original file: ./PLBartForConditionalGeneration__57_forward_180.13/PLBartForConditionalGeneration__57_forward_180.13_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/mf/cmftpyjw4tjuxs4daoi5axm2xcakbl7cyed6xawxmtdjphorizxs.py
# Source Nodes: [bmm_1, dropout, softmax], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# bmm_1 => convert_element_type_9
# dropout => gt, mul_1, mul_2
# softmax => amax, div, exp, sub, sum_1
triton_per_fused__softmax__to_copy_native_dropout_5 = async_compile.triton('triton_per_fused__softmax__to_copy_native_dropout_5', '''
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
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_native_dropout_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_native_dropout_5(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 49152
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.load(in_ptr2 + load_seed_offset)
    tmp15 = r2 + (1024*x3)
    tmp16 = tl.rand(tmp14, (tmp15).to(tl.uint32))
    tmp17 = 0.1
    tmp18 = tmp16 > tmp17
    tmp19 = tmp9 / tmp13
    tmp20 = tmp18.to(tl.float32)
    tmp21 = tmp20 * tmp19
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp18, rmask)
    tl.store(out_ptr4 + (r2 + (1024*x3)), tmp19, rmask)
    tl.store(out_ptr5 + (r2 + (1024*x3)), tmp24, rmask)
''')
