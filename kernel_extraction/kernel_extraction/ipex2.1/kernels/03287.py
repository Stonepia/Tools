

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/bk/cbk5dn72ade7lp7d4mbkt7rvwuvapg6bsjpgw5rtw7ns7ilxuw7a.py
# Source Nodes: [reshape_15, reshape_19], Original ATen: [aten.clone]
# reshape_15 => clone_7
# reshape_19 => clone_8
triton_poi_fused_clone_17 = async_compile.triton('triton_poi_fused_clone_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8847360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 12288) % 60
    x3 = (xindex // 737280)
    x4 = xindex % 12288
    x5 = xindex % 737280
    x0 = xindex % 64
    x1 = (xindex // 64) % 192
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (3*x2) + (186*x3) + (186*((3 + (3*x2) + (x4 // 4096)) // 186)) + (186*((12288 + x4 + (12288*x2)) // 761856)) + (x4 // 4096)), None)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = (64*x3) + (64*((3 + (x5 // 4096)) // 186)) + (64*((12288 + x5) // 761856))
    tmp3 = tmp1 + tmp2
    tmp4 = tl.where(tmp3 < 0, tmp3 + 768, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 768), "index out of bounds: 0 <= tmp4 < 768")
    tmp5 = tl.load(in_ptr1 + (x0 + (64*((tmp4 // 64) % 12)) + (768*(x1 % 64)) + (49152*(tmp4 % 64))), None)
    tmp6 = tl.load(in_ptr2 + (x0 + (64*((tmp4 // 64) % 12)) + (768*(x1 % 64)) + (49152*(tmp4 % 64))), None)
    tl.store(out_ptr0 + (x7), tmp5, None)
    tl.store(out_ptr1 + (x7), tmp6, None)
''')
