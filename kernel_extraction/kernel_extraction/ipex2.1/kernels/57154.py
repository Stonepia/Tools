

# Original file: ./Background_Matting___60.0/Background_Matting___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/i5/ci5qtdclf3reuyq23gnux5befdm5lbwqz366otws4q5jaqpq2zm4.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_14 = async_compile.triton('triton_poi_fused_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp5, tmp5)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')