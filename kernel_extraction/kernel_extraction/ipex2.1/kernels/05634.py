

# Original file: ./detectron2_fasterrcnn_r_50_dc5__42_inference_82.22/detectron2_fasterrcnn_r_50_dc5__42_inference_82.22.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/yj/cyjwsqkx3362yxxhigu37vrgvk7ohycy6bys7vmcuuc4qxao7bov.py
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 80)
    tmp0 = tl.load(in_ptr0 + (1 + (4*x2)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (3 + (4*x1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (1 + (4*x1)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (3 + (4*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 10.0
    tmp3 = tmp1 / tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = 0.5
    tmp9 = tmp6 * tmp8
    tmp10 = tmp5 + tmp9
    tmp11 = tmp7 + tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 5.0
    tmp15 = tmp13 / tmp14
    tmp16 = 4.135166556742356
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tl.exp(tmp17)
    tmp19 = tmp18 * tmp6
    tmp20 = tmp19 * tmp8
    tmp21 = tmp11 - tmp20
    tmp22 = tmp11 + tmp20
    tl.store(out_ptr0 + (4*x2), tmp21, xmask)
    tl.store(out_ptr1 + (4*x2), tmp22, xmask)
''')
