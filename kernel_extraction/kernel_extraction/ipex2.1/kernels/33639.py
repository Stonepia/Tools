

# Original file: ./MobileBertForQuestionAnswering__0_backward_279.1/MobileBertForQuestionAnswering__0_backward_279.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/hz/chzpyjjxgb2ndzreh5ppx7tuew6gcq5eewn2pepfut62za7lewqq.py
# Source Nodes: [cross_entropy], Original ATen: [aten.add, aten.constant_pad_nd, aten.embedding_dense_backward, aten.nll_loss_forward, aten.slice_backward]
# cross_entropy => full_default_3
triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_44 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_44(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 128)
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (128 + x0 + (384*x3)), None).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30522, tmp0)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp0 == tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = x1
    tmp7 = tl.full([1], 127, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = 1 + x1
    tmp10 = tmp9 >= tmp2
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr1 + (640 + x0 + (384*x3)), tmp11, other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.where(tmp11, tmp13, 0.0)
    tmp15 = tl.where(tmp8, tmp14, 0.0)
    tmp16 = 0.0
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tmp5 + tmp17
    tmp19 = tl.full([1], 1, tl.int64)
    tmp20 = tmp6 >= tmp19
    tmp21 = (-1) + x1
    tmp22 = tl.full([1], 128, tl.int64)
    tmp23 = tmp21 < tmp22
    tmp24 = tmp23 & tmp20
    tmp25 = tl.load(in_ptr1 + ((-384) + x0 + (384*x3)), tmp24, other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.where(tmp24, tmp26, 0.0)
    tmp28 = tl.where(tmp20, tmp27, 0.0)
    tmp29 = tl.where(tmp20, tmp28, tmp16)
    tmp30 = tmp18 + tmp29
    tmp31 = tl.where(tmp3, tmp16, tmp30)
    tl.atomic_add(out_ptr0 + (x0 + (128*tmp1)), tmp31, None)
''')
