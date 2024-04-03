

# Original file: ./detectron2_maskrcnn_r_101_fpn__26_inference_66.6/detectron2_maskrcnn_r_101_fpn__26_inference_66.6_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/w2/cw2dgisq32mjsnz34dw5mgyyqqiroxxgqbwkkm7rmfjb5dw6ahxi.py
# Source Nodes: [stack_3], Original ATen: [aten.stack]
# stack_3 => cat_3
triton_poi_fused_stack_7 = async_compile.triton('triton_poi_fused_stack_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_7(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2850
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + (4*x0)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (3 + (4*x0)), xmask)
    tmp5 = tl.load(in_ptr1 + (1 + (4*x0)), xmask)
    tmp12 = tl.load(in_ptr0 + (3 + (4*x0)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 / tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = 0.5
    tmp9 = tmp6 * tmp8
    tmp10 = tmp5 + tmp9
    tmp11 = tmp7 + tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 / tmp2
    tmp15 = 4.135166556742356
    tmp16 = triton_helpers.minimum(tmp14, tmp15)
    tmp17 = tl.exp(tmp16)
    tmp18 = tmp17 * tmp6
    tmp19 = tmp18 * tmp8
    tmp20 = tmp11 - tmp19
    tmp21 = tmp11 + tmp19
    tl.store(out_ptr0 + (4*x0), tmp20, xmask)
    tl.store(out_ptr1 + (4*x0), tmp21, xmask)
''')
