

# Original file: ./detectron2_maskrcnn_r_101_c4__60_inference_100.40/detectron2_maskrcnn_r_101_c4__60_inference_100.40.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/vw/cvwizuta2ko4pzw44jobbuwjjmphpfrk74owjgllmls6yrx6r3ib.py
# Source Nodes: [sigmoid, split_1], Original ATen: [aten.sigmoid, aten.split_with_sizes]
# sigmoid => sigmoid
# split_1 => split_with_sizes
triton_poi_fused_sigmoid_split_with_sizes_1 = async_compile.triton('triton_poi_fused_sigmoid_split_with_sizes_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_split_with_sizes_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_sigmoid_split_with_sizes_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 80, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 80)) | ~xmask, "index out of bounds: 0 <= tmp1 < 80")
    tmp2 = tl.load(in_ptr1 + (tmp1 + (80*x2)), xmask)
    tmp3 = tl.sigmoid(tmp2)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')
