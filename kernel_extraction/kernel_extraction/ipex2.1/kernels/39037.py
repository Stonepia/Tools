

# Original file: ./DebertaV2ForMaskedLM__0_backward_207.1/DebertaV2ForMaskedLM__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/n7/cn7xis2vdechcsuh2t74vl3u37bq6b4uzoem6usiti23psxiwuhj.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_170
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.where(tmp4 < 0, tmp4 + 128100, tmp4)
    # tl.device_assert((0 <= tmp5) & (tmp5 < 128100), "index out of bounds: 0 <= tmp5 < 128100")
    tmp6 = -1.0
    tl.store(out_ptr0 + (tmp5 + (128100*x0)), tmp6, xmask)
''')