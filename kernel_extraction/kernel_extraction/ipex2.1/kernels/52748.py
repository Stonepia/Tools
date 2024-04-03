

# Original file: ./BlenderbotForCausalLM__25_forward_80.3/BlenderbotForCausalLM__25_forward_80.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/xb/cxbn5pttq3vvg6qsmxb2qzw5v623jnm4utyf6qyb52dhuhzmfvst.py
# Source Nodes: [add_1, dropout_1, l__self___fc1, l__self___final_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_dropout, aten.native_layer_norm, aten.view]
# add_1 => add_3
# dropout_1 => gt, mul_3, mul_4
# l__self___fc1 => convert_element_type_12, view_18
# l__self___final_layer_norm => add_4, add_5, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_7 = async_compile.triton('triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_7', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_dropout_native_layer_norm_view_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp7 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (2560*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.1
        tmp5 = tmp3 > tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = 1.1111111111111112
        tmp10 = tmp8 * tmp9
        tmp12 = tmp10.to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
        tl.store(out_ptr1 + (r1 + (2560*x0)), tmp5, rmask & xmask)
        tl.store(in_out_ptr0 + (r1 + (2560*x0)), tmp10, rmask & xmask)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr2 + (x0), tmp15, xmask)
    tmp23_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp23_mean_next, tmp23_m2_next, tmp23_weight_next = triton_helpers.welford_reduce(
            tmp22, tmp23_mean, tmp23_m2, tmp23_weight,
        )
        tmp23_mean = tl.where(rmask & xmask, tmp23_mean_next, tmp23_mean)
        tmp23_m2 = tl.where(rmask & xmask, tmp23_m2_next, tmp23_m2)
        tmp23_weight = tl.where(rmask & xmask, tmp23_weight_next, tmp23_weight)
    tmp23_tmp, tmp24_tmp, tmp25_tmp = triton_helpers.welford(
        tmp23_mean, tmp23_m2, tmp23_weight, 1
    )
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = 2560.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp30, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp31 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 + tmp33
        tmp35 = tmp34 - tmp15
        tmp36 = tmp35 * tmp30
        tmp38 = tmp36 * tmp37
        tmp40 = tmp38 + tmp39
        tmp41 = tmp40.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (2560*x0)), tmp41, rmask & xmask)
''')
