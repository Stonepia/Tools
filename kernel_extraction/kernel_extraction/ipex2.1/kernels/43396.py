

# Original file: ./detectron2_maskrcnn__30_inference_70.10/detectron2_maskrcnn__30_inference_70.10_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/vz/cvzs3vfhkmjptuykbhlv7qfn7dgr7czrpdwbz6fddjpjcosttyrn.py
# Source Nodes: [reshape_1], Original ATen: [aten._unsafe_view, aten.clone]
# reshape_1 => clone_1, view_3
triton_poi_fused__unsafe_view_clone_1 = async_compile.triton('triton_poi_fused__unsafe_view_clone_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_view_clone_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 60800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = (x0 // 304)
    tmp1 = tmp0.to(tl.float64)
    tmp2 = tl.full([1], 4.0, tl.float64)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.full([1], 0.0, tl.float64)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')
