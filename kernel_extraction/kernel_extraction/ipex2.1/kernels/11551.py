

# Original file: ./detectron2_fasterrcnn_r_50_c4__55_inference_95.35/detectron2_fasterrcnn_r_50_c4__55_inference_95.35.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ng/cngg7c2wglwrsehm33qfmywb63qi75jgro5piw4djzb6sxaubsfs.py
# Source Nodes: [stack], Original ATen: [aten.stack]
# stack => cat
triton_poi_fused_stack_2 = async_compile.triton('triton_poi_fused_stack_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_2', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp7 = tl.load(in_ptr0 + (1 + (4*x0)), xmask)
    tmp19 = tl.load(in_ptr0 + (2 + (4*x0)), xmask)
    tmp0 = tl.full([1], 1, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp2 == tmp2
    tmp4 = tmp1 & tmp3
    tmp5 = tl.load(in_ptr0 + (1 + (4*x0)), tmp4 & xmask, other=0.0)
    tmp6 = tl.where(tmp4, tmp5, 0.0)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = 427.0
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp13 >= tmp0
    tmp15 = tmp0 == tmp2
    tmp16 = tmp14 & tmp15
    tmp17 = tl.load(in_ptr0 + (1 + (4*x0)), tmp16 & xmask, other=0.0)
    tmp18 = tl.where(tmp16, tmp17, 0.0)
    tmp20 = tl.where(tmp16, tmp18, tmp19)
    tmp21 = triton_helpers.maximum(tmp20, tmp9)
    tmp22 = 640.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tl.store(out_ptr0 + (4*x0), tmp12, xmask)
    tl.store(out_ptr1 + (4*x0), tmp23, xmask)
''')
