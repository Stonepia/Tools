

# Original file: ./vision_maskrcnn__58_inference_98.38/vision_maskrcnn__58_inference_98.38_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/wq/cwq3lf7il6bdaeyxxwwzop63ypi5ixfm5kk6amjwbshkcmqauaux.py
# Source Nodes: [add_1, clamp_3], Original ATen: [aten.add, aten.clamp]
# add_1 => add_8
# clamp_3 => clamp_max_3
triton_poi_fused_add_clamp_2 = async_compile.triton('triton_poi_fused_add_clamp_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clamp_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (3))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.full([1], 427, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, None)
''')
