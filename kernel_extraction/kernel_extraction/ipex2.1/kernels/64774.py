

# Original file: ./detectron2_maskrcnn_r_101_c4__66_inference_106.46/detectron2_maskrcnn_r_101_c4__66_inference_106.46_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ed/cednf6ql47dddz2acerdyoxh4gbqnixsicbojny3xgwrypyvmxwb.py
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

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17216640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 640) % 427
    x2 = (xindex // 273280)
    x4 = xindex
    tmp8 = tl.load(in_ptr0 + (1 + (4*x2)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (3 + (4*x2)), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 - tmp8
    tmp11 = tmp10 - tmp8
    tmp12 = tmp9 / tmp11
    tmp13 = 2.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 - tmp2
    tl.store(out_ptr0 + (2*x4), tmp15, xmask)
''')
