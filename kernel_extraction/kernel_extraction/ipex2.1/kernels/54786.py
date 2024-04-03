

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/qe/cqe5fspjmahakwk3njdbkw5zjiuivyt5vkaunsotzn7wkeqkkkxq.py
# Source Nodes: [view_7], Original ATen: [aten.view]
# view_7 => view_38
triton_poi_fused_view_13 = async_compile.triton('triton_poi_fused_view_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_view_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp4, tmp3)
    tl.store(out_ptr0 + (x0), tmp5, None)
''')
