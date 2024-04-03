

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/jb/cjb64wuecihb2xchh4cr527sqod6oiziwg4ijqwuvj64mf2vuivr.py
# Source Nodes: [add, l__mod___deberta_encoder_layer_0_attention_output_layer_norm, trampoline_autograd_apply, trampoline_autograd_apply_3], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.rsub]
# add => add_3
# l__mod___deberta_encoder_layer_0_attention_output_layer_norm => add_4, add_5, convert_element_type_14, convert_element_type_15, mul_8, mul_9, rsqrt_1, sub_5, var_mean_1
# trampoline_autograd_apply => full_default_1
# trampoline_autograd_apply_3 => convert_element_type_12, convert_element_type_13, lt_2, mul_7, sub_4, where_4
triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_rsub_6 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_rsub_6', '''
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*i1', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_rsub_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_rsub_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (1536*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = 0.9
        tmp4 = tmp2 < tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 1.0
        tmp7 = tmp6 - tmp5
        tmp8 = (tmp7 != 0)
        tmp10 = 0.0
        tmp11 = tl.where(tmp8, tmp10, tmp9)
        tmp12 = 1.1111111111111112
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight,
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
        tl.store(in_out_ptr0 + (r1 + (1536*x0)), tmp15, rmask & xmask)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight,
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp27 = 1536.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-07
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp31, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp32 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp18
        tmp35 = tmp34 * tmp31
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 * tmp37
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp38 + tmp40
        tmp42 = tmp41.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp42, rmask & xmask)
''')
