

# Original file: ./detectron2_fasterrcnn_r_101_dc5__42_inference_82.22/detectron2_fasterrcnn_r_101_dc5__42_inference_82.22.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/vu/cvuu4ah33jf2y3yimxlz42xruwi3sfgam7lrlase4o4oylvcp5tn.py
# Source Nodes: [stack], Original ATen: [aten.stack]
# stack => cat
triton_poi_fused_stack_0 = async_compile.triton('triton_poi_fused_stack_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 80)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask)
    tmp3 = tl.load(in_ptr1 + (2 + (4*x1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2 + (4*x2)), xmask)
    tmp1 = 10.0
    tmp2 = tmp0 / tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = 0.5
    tmp8 = tmp5 * tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = 5.0
    tmp13 = tmp11 / tmp12
    tmp14 = 4.135166556742356
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tl.exp(tmp15)
    tmp17 = tmp16 * tmp5
    tmp18 = tmp17 * tmp7
    tmp19 = tmp10 - tmp18
    tmp20 = tmp10 + tmp18
    tl.store(out_ptr0 + (4*x2), tmp19, xmask)
    tl.store(out_ptr1 + (4*x2), tmp20, xmask)
''')
