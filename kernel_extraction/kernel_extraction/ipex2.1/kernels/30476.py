

# Original file: ./XLNetLMHeadModel__0_backward_567.1/XLNetLMHeadModel__0_backward_567.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/5t/c5trkru7ybdkusimvsfhk2h23d5aaiq4z54fpst2hzdpywitye6g.py
# Source Nodes: [cross_entropy], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.nll_loss_forward]
# cross_entropy => full_default_1
triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_nll_loss_forward_26 = async_compile.triton('triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_nll_loss_forward_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_nll_loss_forward_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_nll_loss_forward_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp9 = tl.load(in_ptr4 + (x2), None)
    tmp11 = tl.load(in_ptr5 + (x2), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32000, tmp0)
    tmp2 = tl.full([1], -1, tl.int64)
    tmp3 = tmp0 == tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1.1111111111111112
    tmp14 = tmp12 * tmp13
    tmp15 = tmp10 * tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp3, tmp16, tmp15)
    tl.atomic_add(out_ptr0 + (x0 + (1024*tmp1)), tmp17, None)
''')
