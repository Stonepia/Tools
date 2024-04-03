

# Original file: ./detectron2_maskrcnn_r_101_c4__61_inference_101.41/detectron2_maskrcnn_r_101_c4__61_inference_101.41_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/lb/clbmgq7s5ozawuec6rn44omlmrenlc35tatlspnze2kuvkkp7fp3.py
# Source Nodes: [stack], Original ATen: [aten.stack]
# stack => cat
triton_poi_fused_stack_3 = async_compile.triton('triton_poi_fused_stack_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_3', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp8 = tl.load(in_ptr0 + (3 + (4*x0)), xmask)
    tmp0 = tl.full([1], 3, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp3 == tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (3 + (4*x0)), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = 427.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (4*x0), tmp13, xmask)
''')
