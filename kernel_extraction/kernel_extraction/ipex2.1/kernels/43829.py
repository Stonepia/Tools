

# Original file: ./DALLE2_pytorch__27_inference_67.7/DALLE2_pytorch__27_inference_67.7.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6d/c6dknm6wco5v3dmn24tmd6jidlba5kgfadpfmq7dbwu3agfkkdkp.py
# Source Nodes: [add_1, mul_3, mul_4, mul_5, mul_6], Original ATen: [aten.add, aten.mul]
# add_1 => add_1
# mul_3 => mul_3
# mul_4 => mul_4
# mul_5 => mul_5
# mul_6 => mul_6
triton_poi_fused_add_mul_3 = async_compile.triton('triton_poi_fused_add_mul_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 133120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 260
    x2 = (xindex // 8320) % 8
    x3 = (xindex // 66560)
    x5 = xindex % 8320
    x6 = xindex
    x7 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (133120*x3)), None)
    tmp3 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x6), None)
    tmp8 = tl.load(in_ptr3 + (x5), None, eviction_policy='evict_last')
    tmp1 = 16.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp5
    tmp11 = tmp6 + tmp10
    tl.store(out_ptr0 + (x0 + (64*x7)), tmp11, None)
''')
