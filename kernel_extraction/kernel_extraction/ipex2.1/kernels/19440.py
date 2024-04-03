

# Original file: ./DistilBertForMaskedLM__0_forward_97.0/DistilBertForMaskedLM__0_forward_97.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/7e/c7eub6lpif2rq3gevmskbszcnwkuarycpjoxyjnjwvpsq2e3fva3.py
# Source Nodes: [eq, ones, view_3], Original ATen: [aten.eq, aten.ones, aten.view]
# eq => eq
# ones => full
# view_3 => view_12
triton_poi_fused_eq_ones_view_4 = async_compile.triton('triton_poi_fused_eq_ones_view_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*i1', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_ones_view_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_eq_ones_view_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''')
