

# Original file: ./BlenderbotForCausalLM__58_forward_179.14/BlenderbotForCausalLM__58_forward_179.14_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/ba/cba2sr47ldigxf5cx7punbr7p46zxvx3mfkpj3ktxzpvqdvy4tmp.py
# Source Nodes: [add_1, dropout_1, l__self___fc1, l__self___final_layer_norm], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.view]
# add_1 => add_3
# dropout_1 => gt, mul_3, mul_4
# l__self___fc1 => view_18
# l__self___final_layer_norm => add_4, add_5, convert_element_type_4, convert_element_type_5, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
triton_red_fused_add_native_dropout_native_layer_norm_view_5 = async_compile.triton('triton_red_fused_add_native_dropout_native_layer_norm_view_5', '''
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
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*i1', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_dropout_native_layer_norm_view_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_dropout_native_layer_norm_view_5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (2560*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.1
        tmp5 = tmp3 > tmp4
        tmp7 = tmp5.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp12 = tmp6 + tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
        tl.store(out_ptr1 + (r1 + (2560*x0)), tmp5, rmask & xmask)
        tl.store(in_out_ptr0 + (r1 + (2560*x0)), tmp12, rmask & xmask)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr2 + (x0), tmp15, xmask)
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_reduce(
            tmp20, tmp21_mean, tmp21_m2, tmp21_weight,
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = 2560.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp28, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp15
        tmp32 = tmp31 * tmp28
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (2560*x0)), tmp39, rmask & xmask)
''')
