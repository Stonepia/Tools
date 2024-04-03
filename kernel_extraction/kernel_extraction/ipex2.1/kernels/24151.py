

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/7j/c7j4upnob47ezh4m3h2yuri3gxvtxpjbrb7nmv6ponbeq4dcm75x.py
# Source Nodes: [l__self___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
# l__self___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2 => _convolution_pointwise_default_22
triton_poi_fused_convolution_35 = async_compile.triton('triton_poi_fused_convolution_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 784
    x2 = (xindex // 37632)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (720 + x0 + (384*x1) + (301440*x2)), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')