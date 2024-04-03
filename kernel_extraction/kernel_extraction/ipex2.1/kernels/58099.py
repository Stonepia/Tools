

# Original file: ./hf_Bart___60.0/hf_Bart___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/2r/c2rdhvskqf3szxzybxdlmlrazuszbeqxdztwzmhjtwoxxfbppfal.py
# Source Nodes: [add_13, add_18, dropout_24, dropout_27, l__self___model_model_decoder_layers_0_encoder_attn_k_proj, l__self___model_model_decoder_layers_0_encoder_attn_q_proj, l__self___model_model_decoder_layers_0_self_attn_layer_norm, l__self___model_model_encoder_layers_5_final_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_13 => add_43
# add_18 => add_52
# dropout_24 => clone_48
# dropout_27 => clone_55
# l__self___model_model_decoder_layers_0_encoder_attn_k_proj => convert_element_type_135
# l__self___model_model_decoder_layers_0_encoder_attn_q_proj => convert_element_type_132
# l__self___model_model_decoder_layers_0_self_attn_layer_norm => add_53, add_54, mul_55, mul_56, rsqrt_14, sub_21, var_mean_14
# l__self___model_model_encoder_layers_5_final_layer_norm => add_44, add_45, mul_49, mul_50, rsqrt_12, sub_18, var_mean_12
triton_per_fused__to_copy_add_clone_native_layer_norm_9 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_9', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 768, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 + tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp30 / tmp12
    tmp32 = tmp24 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp23 - tmp31
    tmp39 = 768.0
    tmp40 = tmp37 / tmp39
    tmp41 = 1e-05
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp3 - tmp13
    tmp51 = tmp19 / tmp39
    tmp52 = tmp51 + tmp41
    tmp53 = libdevice.rsqrt(tmp52)
    tmp54 = tmp50 * tmp53
    tmp55 = tmp54 * tmp45
    tmp56 = tmp55 + tmp47
    tmp57 = tmp56.to(tl.float32)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp48, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp49, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp56, rmask & xmask)
    tl.store(out_ptr7 + (r1 + (768*x0)), tmp57, rmask & xmask)
''')
