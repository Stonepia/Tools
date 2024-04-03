

# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/ua/cua427yfba6kmm3bf5lt6cm3m5vpijeb3xjq7m54kq5fvidn6njn.py
# Source Nodes: [matmul_3], Original ATen: [aten.clone]
# matmul_3 => clone_26
triton_poi_fused_clone_44 = async_compile.triton('triton_poi_fused_clone_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6569984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 401
    x2 = (xindex // 12832) % 4
    x3 = (xindex // 51328)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (128*x1) + (51328*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')