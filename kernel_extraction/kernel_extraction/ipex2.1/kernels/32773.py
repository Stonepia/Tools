

# Original file: ./DALLE2_pytorch___60.0/DALLE2_pytorch___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/po/cpokrzt6wok2krjcfpvep4pb2ypv4hv3pypkyw735jdxfcnsv7k7.py
# Source Nodes: [getattr_l__self___clip_clip_transformer_resblocks___0___attn], Original ATen: [aten.bmm]
# getattr_l__self___clip_clip_transformer_resblocks___0___attn => bmm_1
triton_poi_fused_bmm_5 = async_compile.triton('triton_poi_fused_bmm_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bmm_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (157696 + (512*(((x0 + (64*(x1 % 8)) + (512*(x1 // 8))) // 512) % 2)) + (1024*(((x0 + (64*(x1 % 8)) + (512*(x1 // 8)) + (1024*x2)) // 1024) % 77)) + ((x0 + (64*(x1 % 8))) % 512)), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''')
