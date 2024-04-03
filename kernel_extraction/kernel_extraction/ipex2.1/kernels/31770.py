

# Original file: ./detectron2_maskrcnn_r_50_c4__66_inference_106.46/detectron2_maskrcnn_r_50_c4__66_inference_106.46_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/5b/c5bfoylvbpqtxvxy7utsfggilig5adkqg7tmwjknshmnryt4vllg.py
# Source Nodes: [ge, setitem, zeros], Original ATen: [aten.ge, aten.index_put, aten.zeros]
# ge => ge_8
# setitem => index_put
# zeros => full_default
triton_poi_fused_ge_index_put_zeros_3 = async_compile.triton('triton_poi_fused_ge_index_put_zeros_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_index_put_zeros_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_ge_index_put_zeros_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9018240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], False, tl.int1)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')
