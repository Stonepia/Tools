

# Original file: ./sam___60.0/sam___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/63/c63er7j2ypb7hc4sclz7qt7mefov5ntltnxta3t2vakadgwnar2p.py
# Source Nodes: [matmul_67], Original ATen: [aten.bmm]
# matmul_67 => bmm_132
triton_poi_fused_bmm_47 = async_compile.triton('triton_poi_fused_bmm_47', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_47', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bmm_47(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp0, xmask)
''')