

# Original file: ./hf_Bart___60.0/hf_Bart___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/s3/cs3oxlpot6zlgnhi54sq5qlglj2h2l4cmlirearfnikuun7yb5y6.py
# Source Nodes: [add, add_1, embedding, l__mod___model_model_encoder_embed_tokens, l__mod___model_model_encoder_layernorm_embedding, mul], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# add_1 => add_1
# embedding => embedding_1
# l__mod___model_model_encoder_embed_tokens => embedding
# l__mod___model_model_encoder_layernorm_embedding => add_2, add_3, convert_element_type, convert_element_type_1, mul_1, mul_2, rsqrt, sub, var_mean
# mul => mul
triton_red_fused_add_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_red_fused_add_embedding_mul_native_layer_norm_0', '''
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_embedding_mul_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 50265, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp1 < 50265")
        tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = 1.0
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight,
        )
        tmp9_mean = tl.where(rmask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(rmask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(rmask & xmask, tmp9_weight_next, tmp9_weight)
    tmp9_tmp, tmp10_tmp, tmp11_tmp = triton_helpers.welford(
        tmp9_mean, tmp9_m2, tmp9_weight, 1
    )
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr2 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.where(tmp0 < 0, tmp0 + 50265, tmp0)
        # tl.device_assert(((0 <= tmp12) & (tmp12 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp12 < 50265")
        tmp13 = tl.load(in_ptr1 + (r1 + (768*tmp12)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = 1.0
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp9
        tmp20 = 768.0
        tmp21 = tmp10 / tmp20
        tmp22 = 1e-05
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp32, rmask & xmask)
''')
