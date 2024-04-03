

# Original file: ./detectron2_maskrcnn_r_50_fpn__25_inference_65.5/detectron2_maskrcnn_r_50_fpn__25_inference_65.5_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/qf/cqf6f7tpy2u4k2xz63cfsuqfpmke6uoljxljbzai5gk742lpxfb2.py
# Source Nodes: [stack_4], Original ATen: [aten.stack]
# stack_4 => cat_4
triton_poi_fused_stack_31 = async_compile.triton('triton_poi_fused_stack_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_31', 'configs': [AttrsDescriptor(divisible_by_16=(), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_31(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 247
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0 % 19
    tmp1 = tmp0.to(tl.float64)
    tmp2 = tl.full([1], 64.0, tl.float64)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.full([1], 0.0, tl.float64)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (4*x0), tmp6, xmask)
''')
