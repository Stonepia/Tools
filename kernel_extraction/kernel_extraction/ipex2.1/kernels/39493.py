

# Original file: ./res2net101_26w_4s___60.0/res2net101_26w_4s___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/iz/cizfwjwf2n2bcr7ksgz7ireasvzx2u7kehbt2cu42jgkjk55joge.py
# Source Nodes: [add_5, cat_61], Original ATen: [aten.add, aten.cat]
# add_5 => add_61
# cat_61 => cat_4
triton_poi_fused_add_cat_12 = async_compile.triton('triton_poi_fused_add_cat_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_cat_12(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 52
    x1 = (xindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (104 + x0 + (208*x1)), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr1 + (x0 + (208*x1)), tmp0, None)
''')