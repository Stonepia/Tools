

# Original file: ./MobileBertForMaskedLM__0_forward_208.0/MobileBertForMaskedLM__0_forward_208.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/4y/c4yuu4fqd7nkpciymvaxwl6gpnh62kablihnl6x67gzyu6yhncdt.py
# Source Nodes: [add, add_1, add_2, l__mod___mobilebert_embeddings_position_embeddings, l__mod___mobilebert_embeddings_token_type_embeddings, l__mod___mobilebert_encoder_layer_0_bottleneck_input_dense, mul_1], Original ATen: [aten.add, aten.embedding, aten.mul, aten.view]
# add => add
# add_1 => add_1
# add_2 => add_2
# l__mod___mobilebert_embeddings_position_embeddings => embedding_1
# l__mod___mobilebert_embeddings_token_type_embeddings => embedding_2
# l__mod___mobilebert_encoder_layer_0_bottleneck_input_dense => view_2
# mul_1 => mul_1
triton_poi_fused_add_embedding_mul_view_3 = async_compile.triton('triton_poi_fused_add_embedding_mul_view_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_view_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_embedding_mul_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 512) % 128
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 512, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 512), "index out of bounds: 0 <= tmp2 < 512")
    tmp3 = tl.load(in_ptr1 + (x0 + (512*tmp2)), None).to(tl.float32)
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tl.store(in_out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr0 + (x3), tmp10, None)
''')
