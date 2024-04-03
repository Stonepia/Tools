

# Original file: ./dlrm___60.0/dlrm___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/3j/c3jythanhsxax7osp5hw3njc6bels524dowbu2odw2ylgtpc4gu4.py
# Source Nodes: [getitem_8], Original ATen: [aten.index]
# getitem_8 => index
triton_poi_fused_index_4 = async_compile.triton('triton_poi_fused_index_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 36
    x1 = (xindex // 36)
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 9, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 9), "index out of bounds: 0 <= tmp1 < 9")
    tmp3 = tl.where(tmp2 < 0, tmp2 + 9, tmp2)
    # tl.device_assert((0 <= tmp3) & (tmp3 < 9), "index out of bounds: 0 <= tmp3 < 9")
    tmp4 = tl.load(in_ptr2 + (tmp3 + (9*tmp1) + (81*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (100*x1)), tmp4, None)
''')
