

# Original file: ./squeezenet1_1___60.0/squeezenet1_1___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/7p/c7poi2aebsmt6bvbdvpstbozxx3cbyupxgtmyide24nhxrs6zaf7.py
# Source Nodes: [l__mod___features_5], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_5 => max_pool2d_with_indices_1
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1492992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 27
    x2 = (xindex // 3456) % 27
    x3 = (xindex // 93312)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (7040 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (14080 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (14208 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (14336 + x0 + (256*x1) + (14080*x2) + (387200*x3)), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')
