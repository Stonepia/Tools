

# Original file: ./sam___60.0/sam___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/bx/cbx3ofjneyzrrlkbja3vobxi3ojvojbuwpb4ilhg7gwiqczgpwd5.py
# Source Nodes: [matmul_64, mul_162, sub_71], Original ATen: [aten._to_copy, aten.mul, aten.sub]
# matmul_64 => convert_element_type_585
# mul_162 => mul_386
# sub_71 => sub_167
triton_poi_fused__to_copy_mul_sub_44 = async_compile.triton('triton_poi_fused__to_copy_mul_sub_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_sub_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_mul_sub_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp2 - tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp5, None)
''')
