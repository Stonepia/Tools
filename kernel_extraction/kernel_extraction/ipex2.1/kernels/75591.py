

# Original file: ./tacotron2__27_inference_67.7/tacotron2__27_inference_67.7_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/wu/cwuy7dqig644z44z3bowbcro3tsr7e535bamjxakdtyiqelsj5jf.py
# Source Nodes: [arange, lt], Original ATen: [aten.arange, aten.lt]
# arange => iota
# lt => lt
triton_poi_fused_arange_lt_0 = async_compile.triton('triton_poi_fused_arange_lt_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_lt_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_arange_lt_0(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0)
    x0 = xindex % ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp1 < tmp0
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''')
