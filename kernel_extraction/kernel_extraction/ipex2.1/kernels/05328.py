

# Original file: ./maml_omniglot___60.0/maml_omniglot___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ot/cotxgv5og2buo57le2uvtqqe4ddtjesto7ckf6dedjrb4gjuepxi.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = -0.0302886962890625
    tmp6 = 0.06561279296875
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1], 3, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.full([1], 4, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = -0.10565185546875
    tmp13 = -0.105712890625
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = 0.0821533203125
    tmp16 = tl.where(tmp9, tmp15, tmp14)
    tmp17 = tl.where(tmp2, tmp7, tmp16)
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''')
