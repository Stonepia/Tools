

# Original file: ./moondream___60.0/moondream___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/6y/c6ykgxj4ywnabf5vds5o5n6kvmskgl34sfljt53fauxpdzoz6dgf.py
# Source Nodes: [add_6, add_7, l__mod___model_embed_tokens, l__mod___model_layers_0_resid_dropout, l__mod___model_layers_0_resid_dropout_1, l__mod___model_layers_1_input_layernorm], Original ATen: [aten.add, aten.clone, aten.embedding, aten.native_layer_norm]
# add_6 => add_8
# add_7 => add_9
# l__mod___model_embed_tokens => embedding
# l__mod___model_layers_0_resid_dropout => clone_3
# l__mod___model_layers_0_resid_dropout_1 => clone_4
# l__mod___model_layers_1_input_layernorm => add_10, add_11, convert_element_type_6, convert_element_type_7, mul_10, mul_11, rsqrt_1, sub_2, var_mean_1
triton_red_fused_add_clone_embedding_native_layer_norm_10 = async_compile.triton('triton_red_fused_add_clone_embedding_native_layer_norm_10', '''
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_embedding_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_embedding_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tl.where(tmp3 < 0, tmp3 + 51200, tmp3)
        # tl.device_assert(((0 <= tmp4) & (tmp4 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp4 < 51200")
        tmp5 = tl.load(in_ptr3 + (r1 + (2048*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp2 + tmp5
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
        tmp12 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(tmp3 < 0, tmp3 + 51200, tmp3)
        # tl.device_assert(((0 <= tmp15) & (tmp15 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp15 < 51200")
        tmp16 = tl.load(in_ptr3 + (r1 + (2048*tmp15)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp14 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp9
        tmp20 = 2048.0
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
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp32, rmask & xmask)
''')
