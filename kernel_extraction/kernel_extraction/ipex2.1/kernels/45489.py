

# Original file: ./doctr_reco_predictor___60.0/doctr_reco_predictor___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/cm/ccmshhn7554cyxss4xop3dr7wku35rikallawym25mkbmm6imxt2.py
# Source Nodes: [l__self___feat_extractor_6], Original ATen: [aten.max_pool2d_with_indices]
# l__self___feat_extractor_6 => max_pool2d_with_indices
triton_poi_fused_max_pool2d_with_indices_1 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x2 = (xindex // 4096)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (16384*x2)), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (16384*x2)), None)
    tmp3 = tl.load(in_ptr0 + (8192 + x0 + (128*x1) + (16384*x2)), None)
    tmp5 = tl.load(in_ptr0 + (8256 + x0 + (128*x1) + (16384*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')