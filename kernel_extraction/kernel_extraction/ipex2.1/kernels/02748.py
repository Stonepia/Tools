

# Original file: ./sebotnet33ts_256___60.0/sebotnet33ts_256___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/dl/cdlmafbc2d7xe7sev3mqcjfhnirjmqzxyzhrresfgbgtrdktzppv.py
# Source Nodes: [reshape_13], Original ATen: [aten.clone]
# reshape_13 => clone_8
triton_poi_fused_clone_35 = async_compile.triton('triton_poi_fused_clone_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (768*x2) + (196608*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''')
