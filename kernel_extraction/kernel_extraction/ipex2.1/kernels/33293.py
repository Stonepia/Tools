

# Original file: ./hf_Whisper___60.0/hf_Whisper___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/rw/crw2foah6og2iem67bvvkujsyhlhcoturckj7qtxmkxoydmdg55r.py
# Source Nodes: [add, add_1, dropout_2, l__self___encoder_layers_0_fc1, l__self___encoder_layers_0_final_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add => add_2
# add_1 => add_5
# dropout_2 => clone_7
# l__self___encoder_layers_0_fc1 => convert_element_type_21
# l__self___encoder_layers_0_final_layer_norm => add_6, add_7, clone_8, mul_10, mul_9, rsqrt_1, sub_2, var_mean_1
triton_red_fused__to_copy_add_clone_native_layer_norm_7 = async_compile.triton('triton_red_fused__to_copy_add_clone_native_layer_norm_7', '''
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
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12000
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1500
    x1 = (xindex // 1500)
    x3 = xindex
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.5
        tmp5 = tmp3 * tmp4
        tmp6 = 0.7071067811865476
        tmp7 = tmp3 * tmp6
        tmp8 = libdevice.erf(tmp7)
        tmp9 = 1.0
        tmp10 = tmp8 + tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp23 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 0.5
        tmp28 = tmp26 * tmp27
        tmp29 = 0.7071067811865476
        tmp30 = tmp26 * tmp29
        tmp31 = libdevice.erf(tmp30)
        tmp32 = 1.0
        tmp33 = tmp31 + tmp32
        tmp34 = tmp28 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp38 = tmp36 + tmp37
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp38 + tmp40
        tmp42 = tmp41 - tmp20
        tmp43 = 384.0
        tmp44 = tmp21 / tmp43
        tmp45 = 1e-05
        tmp46 = tmp44 + tmp45
        tmp47 = libdevice.rsqrt(tmp46)
        tmp48 = tmp42 * tmp47
        tmp50 = tmp48 * tmp49
        tmp52 = tmp50 + tmp51
        tmp53 = tmp52.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp53, rmask & xmask)
''')
