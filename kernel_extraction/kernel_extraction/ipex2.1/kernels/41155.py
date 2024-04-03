

# Original file: ./fastNLP_Bert__21_inference_61.1/fastNLP_Bert__21_inference_61.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/zu/czueucxwzfheibyryypfpie5youchdov6h3fyngmd5hiypbmiy3z.py
# Source Nodes: [add, add_1, l__self___embeddings_layer_norm, l__self___embeddings_position_embeddings, l__self___embeddings_token_type_embeddings, l__self___embeddings_word_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# add => add
# add_1 => add_1
# l__self___embeddings_layer_norm => add_2, add_3, convert_element_type_1, convert_element_type_2, mul_1, mul_2, rsqrt, sub_1, var_mean
# l__self___embeddings_position_embeddings => embedding_1
# l__self___embeddings_token_type_embeddings => embedding_2
# l__self___embeddings_word_embeddings => embedding
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, rnumel):
    xnumel = 475
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 21128, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 21128)) | ~xmask, "index out of bounds: 0 <= tmp1 < 21128")
    tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp6 = tl.where(tmp5 < 0, tmp5 + 2, tmp5)
    # tl.device_assert(((0 <= tmp6) & (tmp6 < 2)) | ~xmask, "index out of bounds: 0 <= tmp6 < 2")
    tmp7 = tl.load(in_ptr4 + (r1 + (768*tmp6)), rmask & xmask, other=0.0).to(tl.float32)
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 768, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 768.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-12
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')
