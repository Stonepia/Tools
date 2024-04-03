

# Original file: ./GPT2ForSequenceClassification__0_backward_135.1/GPT2ForSequenceClassification__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ya/cyapllw5bbk6dbotnjhgoturfn6awgdyvwn2ccf2sywkcefmkfla.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_25
triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_23 = async_compile.triton('triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_23(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (786432 + x2), None)
    tmp5 = tl.load(in_ptr1 + (1572864 + x2), None)
    tmp7 = tl.load(in_ptr1 + (2359296 + x2), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 1024, tmp0)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], False, tl.int1)
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp10, tmp8)
    tl.atomic_add(out_ptr0 + (x0 + (768*tmp1)), tmp11, None)
''')
