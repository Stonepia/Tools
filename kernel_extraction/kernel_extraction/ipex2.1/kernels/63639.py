

# Original file: ./hf_Reformer__21_inference_61.1/hf_Reformer__21_inference_61.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/3q/c3qduyy5ofovng2s7cgc2m3dorghdlq6m4ijedizpv57dgztcxlr.py
# Source Nodes: [l__self___word_embeddings], Original ATen: [aten.embedding]
# l__self___word_embeddings => embedding
triton_poi_fused_embedding_1 = async_compile.triton('triton_poi_fused_embedding_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 320, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 320), "index out of bounds: 0 <= tmp1 < 320")
    tmp2 = tl.load(in_ptr1 + (x0 + (256*tmp1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp2, None)
''')
