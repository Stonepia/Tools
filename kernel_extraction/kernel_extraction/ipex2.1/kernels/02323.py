

# Original file: ./BartForConditionalGeneration__21_backward_388.56/BartForConditionalGeneration__21_backward_388.56.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/ey/ceye3d3zxneoyljzb7d6jyob3z5iv7xi2dgugg74oayb3sm66ftr.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_1 = async_compile.triton('triton_poi_fused_embedding_dense_backward_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 1026, tmp0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.full([1], False, tl.int1)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.atomic_add(out_ptr0 + (x0 + (1024*tmp1)), tmp6, None)
''')
