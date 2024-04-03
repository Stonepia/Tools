

# Original file: ./MegatronBertForCausalLM__0_backward_207.1/MegatronBertForCausalLM__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/nw/cnwubbsslnf7s3a7rkbir2iokpjjic76mytlxwtxt7b46kcxopgo.py
# Source Nodes: [], Original ATen: [aten.nll_loss_backward]

triton_poi_fused_nll_loss_backward_1 = async_compile.triton('triton_poi_fused_nll_loss_backward_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_nll_loss_backward_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2044
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 29056, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 29056)) | ~xmask, "index out of bounds: 0 <= tmp1 < 29056")
    tmp2 = -1.0
    tl.store(out_ptr0 + (tmp1 + (29056*x0)), tmp2, xmask)
''')