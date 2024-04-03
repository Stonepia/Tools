

# Original file: ./MobileBertForMaskedLM__0_backward_210.1/MobileBertForMaskedLM__0_backward_210.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/yp/cyp2nc4kiq2b7wk7k3kehhhwa7zwwswvayfyntspr53ccvnw7iog.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_2
triton_poi_fused_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.where(tmp4 < 0, tmp4 + 30522, tmp4)
    # tl.device_assert((0 <= tmp5) & (tmp5 < 30522), "index out of bounds: 0 <= tmp5 < 30522")
    tmp6 = -1.0
    tl.store(out_ptr0 + (tmp5 + (30522*x0)), tmp6, None)
''')
