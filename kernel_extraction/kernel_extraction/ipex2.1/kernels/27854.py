

# Original file: ./speech_transformer__24_inference_64.4/speech_transformer__24_inference_64.4_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/l2/cl26lc77b5trhoetvfwepud3onxs6g4ipovc6saynrqy6w4ue6ay.py
# Source Nodes: [new_1], Original ATen: [aten.lift_fresh]
# new_1 => full_default_1
triton_poi_fused_lift_fresh_0 = async_compile.triton('triton_poi_fused_lift_fresh_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_lift_fresh_0', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_lift_fresh_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.full([1], 1, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''')
