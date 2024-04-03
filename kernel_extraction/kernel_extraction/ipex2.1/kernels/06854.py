

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ej/cejie73ktre5wvlgtdrxt6qwavvsxuyqktynasoe7jkglfgdeipr.py
# Source Nodes: [add_3, l__self___distilbert_transformer_layer_0_output_layer_norm, l__self___distilbert_transformer_layer_1_ffn_lin1, l__self___distilbert_transformer_layer_1_sa_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_3 => add_10
# l__self___distilbert_transformer_layer_0_output_layer_norm => add_9, mul_14
# l__self___distilbert_transformer_layer_1_ffn_lin1 => convert_element_type_35, view_42
# l__self___distilbert_transformer_layer_1_sa_layer_norm => add_11, add_12, mul_17, mul_18, rsqrt_3, sub_5, var_mean_3
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_15 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_15', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 32768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.full([1], 768, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tmp7 - tmp17
    tmp25 = 768.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-12
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp29 / tmp25
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp30, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp36, None)
''')
