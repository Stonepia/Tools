

# Original file: ./speech_transformer__28_inference_68.8/speech_transformer__28_inference_68.8_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ds/cdskzanfydkggj3at6qk6wxz6kcq6mk4rpp4twhl5n32ocy4nabw.py
# Source Nodes: [add, l__self___tgt_word_emb, mul], Original ATen: [aten.add, aten.embedding, aten.mul]
# add => add
# l__self___tgt_word_emb => embedding
# mul => mul
triton_poi_fused_add_embedding_mul_0 = async_compile.triton('triton_poi_fused_add_embedding_mul_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_embedding_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 512)
    x0 = xindex % 512
    x4 = xindex % 11264
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 1014, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 1014), "index out of bounds: 0 <= tmp1 < 1014")
    tmp2 = tl.load(in_ptr1 + (x0 + (512*tmp1)), None)
    tmp3 = 22.627416997969522
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x5), tmp6, None)
''')