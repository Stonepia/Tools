

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ft/cft7vsykpynzbla5id6rfqykw4xirij4tqjoos2pct5c52uilcwu.py
# Source Nodes: [cat_6834, iadd_1, iadd_2, iadd_3, iadd_4, softmax_4, stack], Original ATen: [aten._softmax, aten.add, aten.cat, aten.stack]
# cat_6834 => cat_21
# iadd_1 => add_11
# iadd_2 => add_18
# iadd_3 => add_25
# iadd_4 => add_32
# softmax_4 => amax_4, div_4, exp_4, sub_4, sum_5
# stack => cat_3428
triton_per_fused__softmax_add_cat_stack_25 = async_compile.triton('triton_per_fused__softmax_add_cat_stack_25', '''
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
    size_hints=[64, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_cat_stack_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_cat_stack_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 168
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (168*x0)), rmask & xmask)
    tmp1 = tl.load(in_ptr1 + (r1 + (168*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r1 + (168*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr2 + (r1 + (168*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r1 + (168*x0)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r1 + (168*x0)), rmask & xmask, other=0.0)
    tmp2 = float("-inf")
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 + tmp14
    tl.store(out_ptr2 + (r1 + (168*x0)), tmp14, rmask & xmask)
    tl.store(in_out_ptr0 + (r1 + (168*x0)), tmp22, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (336*x0)), tmp14, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (336*x0)), tmp22, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (168*x0)), tmp14, rmask & xmask)
''')
