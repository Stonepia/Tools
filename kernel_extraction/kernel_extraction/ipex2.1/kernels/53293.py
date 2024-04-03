

# Original file: ./MobileBertForQuestionAnswering__0_backward_207.1/MobileBertForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/pq/cpql6hcqe4fxtxf5mo6vsiby3k3eoaij2psyzc2xznxhvyfqazpb.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.mul]

triton_poi_fused_add_embedding_dense_backward_mul_34 = async_compile.triton('triton_poi_fused_add_embedding_dense_backward_mul_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*i64', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_mul_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_embedding_dense_backward_mul_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(tmp9 < 0, tmp9 + 2, tmp9)
    tmp11 = tmp8.to(tl.float32)
    tmp12 = tl.full([1], False, tl.int1)
    tmp13 = 0.0
    tmp14 = tl.where(tmp12, tmp13, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp8, None)
    tl.atomic_add(out_ptr0 + (x0 + (512*tmp10)), tmp14, None)
''')
