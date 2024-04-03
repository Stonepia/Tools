

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/5q/c5qe7qr4zc6jr6zdeyjurewr2f4k5uzzpxr5wzwf655h6oikb7f4.py
# Source Nodes: [reshape_26], Original ATen: [aten.clone]
# reshape_26 => clone_55
triton_poi_fused_clone_38 = async_compile.triton('triton_poi_fused_clone_38', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_38', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 60
    x1 = (xindex // 60) % 4
    x2 = (xindex // 240) % 16
    x3 = (xindex // 3840)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (60*x2) + (960*x1) + (3840*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')
