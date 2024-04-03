

# Original file: ./demucs__21_inference_61.1/demucs__21_inference_61.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ot/cotb36hool6wsp6omkkfb3ziik5vt5tttws4hufux4loryzh4h4c.py
# Source Nodes: [l__self___encoder_0_5, l__self___encoder_2_0, l__self___encoder_2_1, l__self___encoder_2_2], Original ATen: [aten.convolution, aten.glu, aten.relu]
# l__self___encoder_0_5 => glu_2
# l__self___encoder_2_0 => convolution_4
# l__self___encoder_2_1 => relu_2
# l__self___encoder_2_2 => convolution_5
triton_poi_fused_convolution_glu_relu_6 = async_compile.triton('triton_poi_fused_convolution_glu_relu_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_glu_relu_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_glu_relu_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12244992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 1530624)
    x3 = xindex % 1530624
    x1 = (xindex // 5979) % 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (3061248*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (1530624 + x3 + (3061248*x2)), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (256 + x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (x4), tmp7, None)
''')