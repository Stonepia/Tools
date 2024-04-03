

# Original file: ./yolov3___60.0/yolov3___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/hr/chrrfc75z64fouetmray7mnksjgp742nwlfxg6vd63d2oqwrhoxq.py
# Source Nodes: [cat_4], Original ATen: [aten.cat]
# cat_4 => cat_3
triton_poi_fused_cat_27 = async_compile.triton('triton_poi_fused_cat_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1566720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 85
    x1 = (xindex // 85) % 2304
    x2 = (xindex // 195840)
    x3 = xindex % 195840
    tmp13 = tl.load(in_ptr0 + (x0 + (85*(x1 // 768)) + (255*(x1 % 768)) + (195840*x2)), None).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp0 < tmp1
    tmp4 = tmp3 & tmp2
    tmp5 = tl.load(in_ptr0 + (x0 + (85*(x1 // 768)) + (255*(x1 % 768)) + (195840*x2)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.where(tmp4, tmp5, 0.0)
    tmp7 = tl.load(in_ptr0 + (x0 + (85*(x1 // 768)) + (255*(x1 % 768)) + (195840*x2)), tmp2, other=0.0).to(tl.float32)
    tmp8 = tl.where(tmp3, tmp6, tmp7)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tl.where(tmp2, tmp9, 0.0)
    tmp11 = tl.load(in_ptr0 + (x0 + (85*(x1 // 768)) + (255*(x1 % 768)) + (195840*x2)), tmp3, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp3, tmp11, 0.0)
    tmp14 = tl.where(tmp3, tmp12, tmp13)
    tmp15 = tl.where(tmp2, tmp10, tmp14)
    tl.store(out_ptr0 + (x3 + (1028160*x2)), tmp15, None)
''')
