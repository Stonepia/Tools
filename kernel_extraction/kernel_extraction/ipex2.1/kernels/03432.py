

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/uk/cukad6amslhr67gtmtwnnhcenovfh6fqv54bnjmvpvpaki462ztp.py
# Source Nodes: [l__mod___cls_predictions_decoder], Original ATen: [aten.view]
# l__mod___cls_predictions_decoder => view_1086
triton_poi_fused_view_32 = async_compile.triton('triton_poi_fused_view_32', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_view_32(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 206266368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')
