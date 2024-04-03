

# Original file: ./detectron2_maskrcnn_r_101_fpn__24_inference_64.4/detectron2_maskrcnn_r_101_fpn__24_inference_64.4_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/c3/cc3oczmjqqnuqlyt46fmw5bvwpynjhkl6oj5zzqo5qrve562b5zq.py
# Source Nodes: [max_pool2d], Original ATen: [aten.max_pool2d_with_indices]
# max_pool2d => getitem
triton_poi_fused_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 19
    x1 = (xindex // 19) % 13
    x2 = (xindex // 247)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (76*x1) + (950*x2)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''')
