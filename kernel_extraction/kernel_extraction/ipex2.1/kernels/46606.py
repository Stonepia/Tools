

# Original file: ./moondream___60.0/moondream___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/yw/cywepsbjrdlueipfuhloxuofzoeqki6l7k7upa2cbrli2r7zvhxr.py
# Source Nodes: [neg], Original ATen: [aten.neg]
# neg => neg
triton_poi_fused_neg_1 = async_compile.triton('triton_poi_fused_neg_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_neg_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_neg_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 512
    x2 = (xindex // 8192)
    x3 = (xindex // 16)
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (64*x2) + (2048*x1)), None).to(tl.float32)
    tmp1 = -tmp0
    tl.store(out_ptr0 + (x0 + (32*x3)), tmp1, None)
''')
