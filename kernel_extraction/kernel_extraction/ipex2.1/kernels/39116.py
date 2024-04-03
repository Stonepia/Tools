

# Original file: ./DebertaV2ForMaskedLM__0_backward_207.1/DebertaV2ForMaskedLM__0_backward_207.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/yt/cyt5byfyu5blwxuiyaecera36ojdqrkvtedryr43wkilov2ntez3.py
# Source Nodes: [trampoline_autograd_apply], Original ATen: [aten.embedding_dense_backward, aten.masked_fill, aten.sum]
# trampoline_autograd_apply => full_default_1
triton_poi_fused_embedding_dense_backward_masked_fill_sum_29 = async_compile.triton('triton_poi_fused_embedding_dense_backward_masked_fill_sum_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_masked_fill_sum_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_masked_fill_sum_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1536)
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (786432 + x2), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 512, tmp0)
    tmp2 = tl.full([1], -1, tl.int64)
    tmp3 = tmp0 == tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp3, tmp7, tmp6)
    tl.atomic_add(out_ptr0 + (x0 + (1536*tmp1)), tmp8, None)
''')
