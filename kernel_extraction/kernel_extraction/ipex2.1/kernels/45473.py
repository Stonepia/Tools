

# Original file: ./doctr_reco_predictor___60.0/doctr_reco_predictor___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/wx/cwxgge5axbwvetc7sc5pzl7b5c2wp7otnofek6c5w5mvfbrj7vq4.py
# Source Nodes: [l__self___feat_extractor_23], Original ATen: [aten.max_pool2d_with_indices]
# l__self___feat_extractor_23 => max_pool2d_with_indices_2
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8192
    x1 = (xindex // 8192)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16384*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (8192 + x0 + (16384*x1)), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (x2), tmp2, None)
''')
