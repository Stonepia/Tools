

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/te/cteujm2fojodjp456xjv5noc7jy6byzkxjt7oo6f2ab4yrfogq4f.py
# Source Nodes: [cat_35, l__mod___cell_stem_0_comb_iter_3_left_act_1], Original ATen: [aten.cat, aten.relu]
# cat_35 => cat
# l__mod___cell_stem_0_comb_iter_3_left_act_1 => relu_9
triton_poi_fused_cat_relu_14 = async_compile.triton('triton_poi_fused_cat_relu_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_relu_14(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5952096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 54
    x2 = (xindex // 54)
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
    tl.store(out_ptr1 + (x1 + (270*x2)), tmp0, xmask)
''')
