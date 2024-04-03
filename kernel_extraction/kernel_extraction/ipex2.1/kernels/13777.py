

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/u6/cu6e2zhtoiajnu32flnpx73drjcnmhkyula2l3vniquenhpznlnz.py
# Source Nodes: [cat_6850, iadd, invert, softmax, stack], Original ATen: [aten._softmax, aten.add, aten.bitwise_not, aten.cat, aten.stack]
# cat_6850 => cat_5
# iadd => convert_element_type_10
# invert => bitwise_not
# softmax => amax, convert_element_type_9, div, exp, sub, sum_1
# stack => cat_3428
triton_per_fused__softmax_add_bitwise_not_cat_stack_6 = async_compile.triton('triton_per_fused__softmax_add_bitwise_not_cat_stack_6', '''
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
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*i1', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_bitwise_not_cat_stack_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_bitwise_not_cat_stack_6(in_ptr0, in_ptr1, out_ptr0, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r1 + (168*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0 == 0
    tmp3 = float("-inf")
    tmp4 = tl.where(tmp1, tmp3, tmp2)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, float("-inf"))
    tmp9 = triton_helpers.max2(tmp8, 1)[:, None]
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp11 / tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (168*x0)), tmp1, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (168*x0)), tmp17, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (336*x0)), tmp17, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (336*x0)), tmp17, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (168*x0)), tmp17, rmask & xmask)
''')
