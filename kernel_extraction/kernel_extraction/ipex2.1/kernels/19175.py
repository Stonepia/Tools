

# Original file: ./MobileBertForQuestionAnswering__0_forward_277.0/MobileBertForQuestionAnswering__0_forward_277.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/xv/cxvoht645xrw3usnqcstofw6dnmzv2mb676uxojqhyl47qr3nc2l.py
# Source Nodes: [add, add_1, add_2, l__self___mobilebert_embeddings_token_type_embeddings, l__self___mobilebert_encoder_layer_0_bottleneck_input_dense, mul_1], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.view]
# add => add
# add_1 => add_1
# add_2 => add_2
# l__self___mobilebert_embeddings_token_type_embeddings => embedding_2
# l__self___mobilebert_encoder_layer_0_bottleneck_input_dense => convert_element_type_3, view_2
# mul_1 => mul_1
triton_poi_fused__to_copy_add_embedding_mul_view_7 = async_compile.triton('triton_poi_fused__to_copy_add_embedding_mul_view_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_embedding_mul_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_embedding_mul_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex
    x5 = xindex % 65536
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x5), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 + tmp0
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
    tl.store(out_ptr1 + (x2), tmp9, None)
    tl.store(out_ptr2 + (x2), tmp10, None)
''')
