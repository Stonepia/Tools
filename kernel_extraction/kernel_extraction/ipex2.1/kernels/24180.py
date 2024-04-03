

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/w6/cw6v4atgfdunjjygb2a3pcp52edsvjaqpjwwd3ed3mslpbaauq5s.py
# Source Nodes: [cat_26], Original ATen: [aten.cat]
# cat_26 => cat_13
triton_poi_fused_cat_64 = async_compile.triton('triton_poi_fused_cat_64', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_64(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 320
    x1 = (xindex // 320)
    tmp0 = tl.load(in_ptr0 + (x0 + (63040*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (63040*x1)), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 + (63040*x1)), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tl.store(out_ptr0 + (x0 + (63040*x1)), tmp6, None)
''')
