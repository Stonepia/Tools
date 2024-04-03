

# Original file: ./demucs__21_inference_61.1/demucs__21_inference_61.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6f/c6ftsfx6x7aacqq2pcl7g3ntwzhvy66gfc4aktr3vonqnnji5w2k.py
# Source Nodes: [l__self___encoder_0_4, l__self___encoder_1_0, l__self___encoder_1_1, l__self___encoder_1_2], Original ATen: [aten.convolution, aten.glu, aten.relu]
# l__self___encoder_0_4 => glu_1
# l__self___encoder_1_0 => convolution_2
# l__self___encoder_1_1 => relu_1
# l__self___encoder_1_2 => convolution_3
triton_poi_fused_convolution_glu_relu_3 = async_compile.triton('triton_poi_fused_convolution_glu_relu_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_glu_relu_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_glu_relu_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24497152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3062144)
    x3 = xindex % 3062144
    x1 = (xindex // 23923) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (6124288*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (3062144 + x3 + (6124288*x2)), xmask)
    tmp4 = tl.load(in_ptr1 + (128 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (x4), tmp7, xmask)
''')
