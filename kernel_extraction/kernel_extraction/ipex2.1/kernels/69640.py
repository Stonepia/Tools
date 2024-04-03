

# Original file: ./tacotron2___60.0/tacotron2___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ug/cugjw7n6mceaukpfv3qk2q5p42t5dhvcclknxxdcmdi4s53nb2vb.py
# Source Nodes: [l__self___embedding, transpose], Original ATen: [aten.embedding, aten.transpose]
# l__self___embedding => embedding
# transpose => permute
triton_poi_fused_embedding_transpose_0 = async_compile.triton('triton_poi_fused_embedding_transpose_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_transpose_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_transpose_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5505024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x0 = xindex % 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 148, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 148), "index out of bounds: 0 <= tmp1 < 148")
    tmp2 = tl.load(in_ptr1 + (x0 + (512*tmp1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp2, None)
''')