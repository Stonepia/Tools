

# Original file: ./res2net101_26w_4s___60.0/res2net101_26w_4s___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/5c/c5csig22rxoz6jsxkzm7twg7wxgvqlxficxgfssh5za3rwzfah7d.py
# Source Nodes: [add_55, cat_34], Original ATen: [aten.add, aten.cat]
# add_55 => add_412
# cat_34 => cat_31
triton_poi_fused_add_cat_23 = async_compile.triton('triton_poi_fused_add_cat_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_cat_23(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 208
    x1 = (xindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (416 + x0 + (832*x1)), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr1 + (x0 + (832*x1)), tmp0, xmask)
''')
