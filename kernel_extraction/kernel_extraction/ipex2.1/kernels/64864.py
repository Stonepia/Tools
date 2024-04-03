

# Original file: ./demucs__23_inference_63.3/demucs__23_inference_63.3_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/4b/c4bmwp5akjrrrzqogs5mulhfdwhenrgewvpl7fehtpjgudrwhpgn.py
# Source Nodes: [add, l__self___decoder_0_0, l__self___decoder_0_1], Original ATen: [aten.add, aten.convolution, aten.glu]
# add => add
# l__self___decoder_0_0 => convolution
# l__self___decoder_0_1 => glu
triton_poi_fused_add_convolution_glu_1 = async_compile.triton('triton_poi_fused_add_convolution_glu_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_glu_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_convolution_glu_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1474560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 184320)
    x3 = xindex % 184320
    x1 = (xindex // 90) % 2048
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (368640*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (184320 + x3 + (368640*x2)), None)
    tmp4 = tl.load(in_ptr1 + (2048 + x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (x4), tmp7, None)
''')
