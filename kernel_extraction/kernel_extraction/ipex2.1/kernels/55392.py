

# Original file: ./detectron2_maskrcnn__31_inference_71.11/detectron2_maskrcnn__31_inference_71.11.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/lf/clfblgdpuvkdiummrbpj5gzi777b5vviqiddi67ii52timukoa5s.py
# Source Nodes: [reshape], Original ATen: [aten._unsafe_view, aten.clone]
# reshape => clone, view_2
triton_poi_fused__unsafe_view_clone_0 = async_compile.triton('triton_poi_fused__unsafe_view_clone_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_0', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_view_clone_0(out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0 % ks0
    tmp1 = tmp0.to(tl.float64)
    tmp2 = ks1
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.full([1], 0.0, tl.float64)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')
