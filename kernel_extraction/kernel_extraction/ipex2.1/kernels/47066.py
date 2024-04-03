

# Original file: ./volo_d1_224___60.0/volo_d1_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/ua/cuagvd3bar342o4eg2dai4dixjr4epoootzgjxofjvdx6lzjyv3k.py
# Source Nodes: [getattr_l__mod___network_0___0___attn_attn], Original ATen: [aten.addmm]
# getattr_l__mod___network_0___0___attn_attn => _linear_pointwise_default_86
triton_poi_fused_addmm_3 = async_compile.triton('triton_poi_fused_addmm_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_addmm_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12544
    x1 = (xindex // 12544)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((196*x1) + (37632*(x0 // 196)) + (x0 % 196)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')
