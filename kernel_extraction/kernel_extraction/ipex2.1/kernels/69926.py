

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ln/clntp3oi2p3bs3uev2vmp3ndoaqiupkwjkcovbjcwg246qlttcnx.py
# Source Nodes: [l__mod___encoder_b_encoder], Original ATen: [aten.bmm]
# l__mod___encoder_b_encoder => bmm_25
triton_poi_fused_bmm_21 = async_compile.triton('triton_poi_fused_bmm_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bmm_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1261568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 256
    x2 = (xindex // 16384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2523136 + (512*(((x0 + (64*(x1 % 8)) + (512*(x1 // 8))) // 512) % 32)) + (16384*(((x0 + (64*(x1 % 8)) + (512*(x1 // 8)) + (16384*x2)) // 16384) % 77)) + ((x0 + (64*(x1 % 8))) % 512)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')