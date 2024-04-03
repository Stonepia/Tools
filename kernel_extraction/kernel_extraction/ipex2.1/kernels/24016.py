

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/jq/cjqsljlzas2g5vrhi7n2zijxnrhlzf5e5j2nzxjgzefga2gf5ikf.py
# Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
# l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0 => _convolution_pointwise_default_6
triton_poi_fused_convolution_69 = async_compile.triton('triton_poi_fused_convolution_69', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_69(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 49
    x2 = (xindex // 6272)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2560 + x0 + (1536*x1) + (76800*x2)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')
