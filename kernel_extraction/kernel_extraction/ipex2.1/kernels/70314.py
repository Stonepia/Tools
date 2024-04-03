

# Original file: ./MobileBertForMaskedLM__0_backward_210.1/MobileBertForMaskedLM__0_backward_210.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/ep/ceplnvf2madkrlf6irlcuc3ecxcqi4y2phqeqn5gueq6fopm27ho.py
# Source Nodes: [], Original ATen: [aten.add, aten.constant_pad_nd, aten.embedding_dense_backward, aten.slice_backward]

triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_slice_backward_42 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_slice_backward_42', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_slice_backward_42', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_slice_backward_42(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = x1
    tmp6 = tl.full([1], 127, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = 1 + x1
    tmp9 = tmp8 >= tmp2
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr1 + (640 + x0 + (384*x3)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = tl.where(tmp7, tmp12, 0.0)
    tmp14 = 0.0
    tmp15 = tl.where(tmp7, tmp13, tmp14)
    tmp16 = tmp4 + tmp15
    tmp17 = tl.full([1], 1, tl.int64)
    tmp18 = tmp5 >= tmp17
    tmp19 = (-1) + x1
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = tmp21 & tmp18
    tmp23 = tl.load(in_ptr1 + ((-384) + x0 + (384*x3)), tmp22, other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp22, tmp23, 0.0)
    tmp25 = tl.where(tmp18, tmp24, 0.0)
    tmp26 = tl.where(tmp18, tmp25, tmp14)
    tmp27 = tmp16 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.where(tmp3, tmp14, tmp28)
    tl.atomic_add(out_ptr0 + (x0 + (128*tmp1)), tmp29, None)
''')
