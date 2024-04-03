

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/i3/ci3te2lkqynrpetjjuu4t576b4dioap23irw4w7sk3yvxluk2kxc.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_3
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 784
    x2 = (xindex // 37632) % 16
    x3 = (xindex // 602112)
    x4 = (xindex // 37632)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (48*x2) + (2304*x1) + (1806336*x3)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (48*x4)), None, eviction_policy='evict_last')
    tmp2 = tl.sqrt(tmp1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1e-12
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp0 / tmp5
    tl.store(out_ptr0 + (x5), tmp6, None)
''')
