

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/5p/c5pmg3cun73sldj3ublomcwerfhulkalwqasqw4fi2o7azqxhxzz.py
# Source Nodes: [add, iadd, l__self___bert_embeddings_layer_norm, l__self___bert_embeddings_word_embeddings, l__self___bert_encoder_layer_0_attention_self_query], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.native_layer_norm]
# add => add
# iadd => add_1
# l__self___bert_embeddings_layer_norm => add_2, add_3, mul_1, mul_2, rsqrt, sub, var_mean
# l__self___bert_embeddings_word_embeddings => embedding
# l__self___bert_encoder_layer_0_attention_self_query => convert_element_type
triton_red_fused__to_copy_add_embedding_native_layer_norm_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_native_layer_norm_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 50358, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 50358), "index out of bounds: 0 <= tmp1 < 50358")
        tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.where(tmp0 < 0, tmp0 + 50358, tmp0)
        # tl.device_assert((0 <= tmp11) & (tmp11 < 50358), "index out of bounds: 0 <= tmp11 < 50358")
        tmp12 = tl.load(in_ptr1 + (r1 + (768*tmp11)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = 768.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-12
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask)
        tl.store(out_ptr3 + (r1 + (768*x0)), tmp28, rmask)
''')
