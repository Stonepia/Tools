

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/yu/cyuwnjuepzubt2om2qa4kklrxffindtaid6i5wfzb34t2bqpqaon.py
# Source Nodes: [matmul_7], Original ATen: [aten.clone]
# matmul_7 => clone_17
triton_poi_fused_clone_37 = async_compile.triton('triton_poi_fused_clone_37', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37748736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 144
    x2 = (xindex // 9216) % 4
    x0 = xindex % 64
    x3 = (xindex // 36864)
    x4 = xindex
    tmp0 = (-2) + (8*((x1 + (144*x2)) // 288)) + (x1 // 12)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (8*(x2 % 2)) + (x1 % 12)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-21760) + (640*(x1 % 12)) + (5120*(x2 % 2)) + (10240*(x1 // 12)) + (81920*((x1 + (144*x2)) // 288)) + (163840*((9216 + x1 + (144*x2) + (576*x0) + (46080*x3)) // 368640)) + (((9216 + x1 + (144*x2) + (576*x0) + (46080*x3)) // 576) % 640)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tl.store(out_ptr0 + (x4), tmp12, None)
''')
