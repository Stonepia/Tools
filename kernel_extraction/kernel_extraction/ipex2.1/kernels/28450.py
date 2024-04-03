

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/du/cdusgfq7nxlibqxbtvc2towudxrtidshm37qnvxowor6h3jmoxth.py
# Source Nodes: [einsum_8], Original ATen: [aten.clone, aten.transpose]
# einsum_8 => clone_2
triton_poi_fused_clone_transpose_8 = async_compile.triton('triton_poi_fused_clone_transpose_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 1024
    x2 = (xindex // 1024)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (1024*x1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (16384*y0)), tmp0, ymask)
    tl.store(out_ptr1 + (x3 + (16384*y0)), tmp0, ymask)
''')
