

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/oe/coefu3qqv2d527bcipnjbtikzjko574bv6dwj6l74mc26kkexait.py
# Source Nodes: [l__mod___encoder_a_encoder], Original ATen: [aten.bmm]
# l__mod___encoder_a_encoder => bmm_1
triton_poi_fused_bmm_8 = async_compile.triton('triton_poi_fused_bmm_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bmm_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1228800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 384
    x2 = (xindex // 24576)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2457600 + (768*(((x0 + (64*(x1 % 12)) + (768*(x1 // 12))) // 768) % 32)) + (24576*(((x0 + (64*(x1 % 12)) + (768*(x1 // 12)) + (24576*x2)) // 24576) % 50)) + ((x0 + (64*(x1 % 12))) % 768)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')
