

# Original file: ./M2M100ForConditionalGeneration__101_forward_310.23/M2M100ForConditionalGeneration__101_forward_310.23_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/wr/cwra35yinuq6kio2iqhychzhsdfci35v5g6xzk6vakpxivq2tgtm.py
# Source Nodes: [dropout_2, softmax_1], Original ATen: [aten._softmax, aten.native_dropout]
# dropout_2 => gt_2, mul_10, mul_11
# softmax_1 => amax_1, div_1, exp_1, sub_3, sum_2
triton_per_fused__softmax_native_dropout_6 = async_compile.triton('triton_per_fused__softmax_native_dropout_6', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_native_dropout_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_native_dropout_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr1 + load_seed_offset)
    tmp12 = r1 + (128*x0)
    tmp13 = tl.rand(tmp11, (tmp12).to(tl.uint32))
    tmp14 = 0.1
    tmp15 = tmp13 > tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp6 / tmp10
    tmp18 = tmp16 * tmp17
    tmp19 = 1.1111111111111112
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp15, rmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp20, rmask)
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''')
