

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/by/cby775p4py35j3njvcfapwlbcvmq2drjj2kymkt3y7kz7wlfpdi4.py
# Source Nodes: [matmul_10], Original ATen: [aten.clone]
# matmul_10 => clone_21
triton_poi_fused_clone_45 = async_compile.triton('triton_poi_fused_clone_45', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_45', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 8
    x3 = (xindex // 1024)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*x2) + (1024*x1) + (8192*((x2 + (8*x1) + (64*x0) + (1024*x3)) // 8192)) + (((x2 + (8*x1) + (64*x0) + (1024*x3)) // 64) % 128)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')
