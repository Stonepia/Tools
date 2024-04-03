

# Original file: ./detectron2_fasterrcnn_r_50_c4__45_inference_85.25/detectron2_fasterrcnn_r_50_c4__45_inference_85.25_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ua/cuaffrctp6x55c6yy6ksngab2x6rhxachuq5xdl4a5os6r6r5llq.py
# Source Nodes: [gt], Original ATen: [aten.gt]
# gt => gt
triton_poi_fused_gt_0 = async_compile.triton('triton_poi_fused_gt_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gt_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_gt_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 79280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 80
    x1 = (xindex // 80)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (81*x1)), xmask).to(tl.float32)
    tmp1 = 0.05
    tmp2 = tmp0 > tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''')