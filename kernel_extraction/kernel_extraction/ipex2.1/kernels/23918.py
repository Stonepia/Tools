

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/ts/ctsl67tuvwhlndivevzncxptj3tkblbe3uhjyaxkvfp3gwazd36e.py
# Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
# l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2 => _convolution_pointwise_default_13
triton_poi_fused_convolution_51 = async_compile.triton('triton_poi_fused_convolution_51', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 120
    x1 = (xindex // 120) % 196
    x2 = (xindex // 23520)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1800 + x0 + (960*x1) + (189120*x2)), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')
