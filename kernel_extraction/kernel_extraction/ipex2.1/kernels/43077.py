

# Original file: ./DALLE2_pytorch__25_inference_65.5/DALLE2_pytorch__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/k5/ck5kdkuncrpznvv733vpwbdfdxrv245jwyva54nvohczc2vsyjuq.py
# Source Nodes: [cat_4], Original ATen: [aten.cat]
# cat_4 => cat_5
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x3 = (xindex // 64)
    x2 = (xindex // 16640)
    x4 = xindex % 16640
    tmp0 = tl.load(in_ptr0 + (64 + x0 + (128*x3)), xmask)
    tl.store(out_ptr0 + (x4 + (16704*x2)), tmp0, xmask)
''')
