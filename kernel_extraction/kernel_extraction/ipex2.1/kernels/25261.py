

# Original file: ./detectron2_maskrcnn_r_101_c4__61_inference_101.41/detectron2_maskrcnn_r_101_c4__61_inference_101.41_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/pr/cprifl2fnmcszgy2rpfn7bssrsdvof7fwttb4allufsw3t2zu2tl.py
# Source Nodes: [stack], Original ATen: [aten.stack]
# stack => cat
triton_poi_fused_stack_1 = async_compile.triton('triton_poi_fused_stack_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp7 = tl.load(in_ptr0 + (4*x0), xmask)
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp1 == tmp0
    tmp4 = tmp2 & tmp3
    tmp5 = tl.load(in_ptr0 + ((-1) + (4*x0)), tmp4 & xmask, other=0.0)
    tmp6 = tl.where(tmp4, tmp5, 0.0)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = 640.0
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (4*x0), tmp12, xmask)
''')
