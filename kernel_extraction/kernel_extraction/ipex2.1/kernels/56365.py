

# Original file: ./detectron2_fasterrcnn_r_101_fpn__25_inference_65.5/detectron2_fasterrcnn_r_101_fpn__25_inference_65.5_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ks/cksqtfs4f7mhrlbm2bek3hln3rysw5b5ziqq3b5liogjh5mbelkg.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_8(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2850
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tl.store(in_out_ptr0 + (x0), tmp0, xmask)
''')
