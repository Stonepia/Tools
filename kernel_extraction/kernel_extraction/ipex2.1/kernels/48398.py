

# Original file: ./deit_base_distilled_patch16_224___60.0/deit_base_distilled_patch16_224___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/v5/cv5b6l5i7bdk7kbxp5hh33s2ogrjh53fj4cye4v2ngw7t4m23ukd.py
# Source Nodes: [reshape_1], Original ATen: [aten.clone]
# reshape_1 => clone_4
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9732096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 198
    x3 = (xindex // 152064)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (12672*x1) + (152064*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')
