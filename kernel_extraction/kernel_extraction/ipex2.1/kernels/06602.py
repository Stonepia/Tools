

# Original file: ./hf_T5_base___60.0/hf_T5_base___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/6i/c6inh3df3ivylekcldnbx6pc46jdd5luw3vt5t5mgpbgplfes26c.py
# Source Nodes: [add_4, add_6, clamp, l__mod___model_encoder_block_0_layer_0_dropout, l__mod___model_encoder_block_0_layer__1__dropout, l__mod___model_encoder_embed_tokens, neg, where_1], Original ATen: [aten.add, aten.clamp, aten.clone, aten.embedding, aten.neg, aten.scalar_tensor, aten.where]
# add_4 => add_6
# add_6 => add_8
# clamp => clamp_max, clamp_min, convert_element_type_8, convert_element_type_9
# l__mod___model_encoder_block_0_layer_0_dropout => clone_3
# l__mod___model_encoder_block_0_layer__1__dropout => clone_5
# l__mod___model_encoder_embed_tokens => embedding
# neg => neg
# where_1 => full_default_2, full_default_3, where_1
triton_poi_fused_add_clamp_clone_embedding_neg_scalar_tensor_where_10 = async_compile.triton('triton_poi_fused_add_clamp_clone_embedding_neg_scalar_tensor_where_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_clone_embedding_neg_scalar_tensor_where_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clamp_clone_embedding_neg_scalar_tensor_where_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x0 = xindex % 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp15 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (x0 + (768*tmp1)), None).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = 64504.0
    tmp9 = 65504.0
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = -tmp10
    tmp12 = triton_helpers.maximum(tmp5, tmp11)
    tmp13 = triton_helpers.minimum(tmp12, tmp10)
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 + tmp15
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''')
