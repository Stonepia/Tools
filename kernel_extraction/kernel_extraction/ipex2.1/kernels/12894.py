

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/p4/cp4df2v4ksry2rdhhl2mewcft2vo4qjmscz3vnizqvp33tucgy6k.py
# Source Nodes: [cat_6846, softmax_1, stack], Original ATen: [aten._softmax, aten.cat, aten.stack]
# cat_6846 => cat_9
# softmax_1 => amax_1, convert_element_type_2, convert_element_type_3, div_1, exp_1, sub_1, sum_2
# stack => cat_3428
triton_per_fused__softmax_cat_stack_13 = async_compile.triton('triton_per_fused__softmax_cat_stack_13', '''
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
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_stack_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_cat_stack_13(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (168*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (r1 + (168*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = float("-inf")
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17 + tmp16
    tl.store(out_ptr2 + (r1 + (168*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (336*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (336*x0)), tmp18, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (168*x0)), tmp16, rmask & xmask)
''')
