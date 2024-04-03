

# Original file: ./detectron2_maskrcnn_r_50_c4__61_inference_101.41/detectron2_maskrcnn_r_50_c4__61_inference_101.41_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/mj/cmjyli3kmu7sicix747fe2qdzi4fu46aao42wscudeuo5ookqoo5.py
# Source Nodes: [and_, gt, gt_1, sub, sub_1], Original ATen: [aten.bitwise_and, aten.gt, aten.sub]
# and_ => bitwise_and
# gt => gt
# gt_1 => gt_1
# sub => sub
# sub_1 => sub_1
triton_poi_fused_bitwise_and_gt_sub_4 = async_compile.triton('triton_poi_fused_bitwise_and_gt_sub_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_gt_sub_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bitwise_and_gt_sub_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + (4*x0)), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3 + (4*x0)), xmask)
    tmp6 = tl.load(in_ptr0 + (1 + (4*x0)), xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7 > tmp3
    tmp9 = tmp4 & tmp8
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')
