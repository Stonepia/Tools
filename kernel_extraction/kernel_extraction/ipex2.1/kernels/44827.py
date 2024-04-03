

# Original file: ./ElectraForCausalLM__0_backward_171.1/ElectraForCausalLM__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/be/cbeyj6bs35t5tpph3iz7umhwz33a53fdhuxfunb5rbgpheddien2.py
# Source Nodes: [l__self___electra_encoder_layer_11_attention_output_layer_norm], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__self___electra_encoder_layer_11_attention_output_layer_norm => convert_element_type_221
triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21 = async_compile.triton('triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21', '''
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
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (r1 + (256*x0)), rmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp5 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 256.0
    tmp22 = tmp14 / tmp21
    tmp23 = tmp5 * tmp21
    tmp24 = tmp23 - tmp9
    tmp25 = tmp15 * tmp20
    tmp26 = tmp24 - tmp25
    tmp27 = tmp22 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 1.1111111111111112
    tmp32 = tmp30 * tmp31
    tmp33 = tmp28 * tmp32
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, rmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp33, rmask)
''')
