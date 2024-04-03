

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/te/ctefgoygmiczlqqojrje4enae35tqr3laurua2wntyszwk2yyh2n.py
# Source Nodes: [iadd_853, iadd_854, iadd_855, iadd_856, softmax_856, stack], Original ATen: [aten._softmax, aten.add, aten.detach, aten.stack]
# iadd_853 => add_5975
# iadd_854 => add_5982
# iadd_855 => add_5989
# iadd_856 => add_5996, alias_3448
# softmax_856 => amax_856, convert_element_type_15417, convert_element_type_15418, div_856, exp_856, sub_856, sum_857
# stack => cat_3428
triton_per_fused__softmax_add_detach_stack_880 = async_compile.triton('triton_per_fused__softmax_add_detach_stack_880', '''
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_detach_stack_880', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_detach_stack_880(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr3 + (r1 + (168*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (r1 + (168*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_out_ptr0 + (r1 + (168*x0)), rmask & xmask, other=0.0).to(tl.float32)
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
    tmp21 = tmp20 + tmp17
    tmp22 = tmp21 + tmp18
    tmp23 = tmp22 + tmp19
    tmp24 = tmp23 + tmp16
    tl.store(out_ptr2 + (r1 + (168*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (168*x0)), tmp17, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (168*x0)), tmp18, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (168*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (168*x0)), tmp16, rmask & xmask)
    tl.store(in_out_ptr0 + (r1 + (168*x0)), tmp24, rmask & xmask)
''')