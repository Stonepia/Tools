

# Original file: ./mixnet_l___60.0/mixnet_l___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/cl/ccly63huojjucy5sbs6namvkdgqhkhonx6thmqm2v3h5bqf7ddwx.py
# Source Nodes: [cat_46], Original ATen: [aten.cat]
# cat_46 => cat_35
triton_poi_fused_cat_103 = async_compile.triton('triton_poi_fused_cat_103', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_103', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_103(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2483712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 396
    x1 = (xindex // 396)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (1584*x1)), tmp0, xmask)
''')