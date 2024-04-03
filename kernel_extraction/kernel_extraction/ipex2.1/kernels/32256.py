

# Original file: ./vision_maskrcnn__30_inference_70.10/vision_maskrcnn__30_inference_70.10.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/eu/ceu4s7ry2l5nm6pnedzaj367g4n4dptlvstvavbx3yekkjbhc4sk.py
# Source Nodes: [add_1], Original ATen: [aten.add]
# add_1 => add_1
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18964
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''')
