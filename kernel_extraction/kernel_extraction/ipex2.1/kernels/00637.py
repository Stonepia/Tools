

# Original file: ./AlbertForMaskedLM__0_backward_207.1/AlbertForMaskedLM__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/2d/c2detmhdxc4t3iaujq3o6clh6zuw7jg4lgyxyvrehgnpw7vaxhbg.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_2
triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_30 = async_compile.triton('triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_30', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_nll_loss_forward_sum_30(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (65536 + x2), None)
    tmp7 = tl.load(in_ptr1 + (131072 + x2), None)
    tmp9 = tl.load(in_ptr1 + (196608 + x2), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 512, tmp0)
    tmp2 = tl.full([1], -1, tl.int64)
    tmp3 = tmp0 == tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tl.atomic_add(out_ptr0 + (x0 + (128*tmp1)), tmp12, None)
''')
