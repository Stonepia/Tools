

# Original file: ./PegasusForConditionalGeneration__60_forward_187.16/PegasusForConditionalGeneration__60_forward_187.16.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/zj/czjlnaphkkivamija6qsucsts3vtpn4tojkyiqsqjxr2owywxaie.py
# Source Nodes: [add_1, dropout_1, l__self___encoder_attn_layer_norm, l__self___encoder_attn_q_proj], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.view]
# add_1 => add_3
# dropout_1 => gt, mul_3, mul_4
# l__self___encoder_attn_layer_norm => add_4, add_5, convert_element_type_4, convert_element_type_5, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
# l__self___encoder_attn_q_proj => view_18
triton_per_fused_add_native_dropout_native_layer_norm_view_5 = async_compile.triton('triton_per_fused_add_native_dropout_native_layer_norm_view_5', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*i1', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_view_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_view_5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (1024*x0)
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
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tl.full([1], 1024, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = 1024.0
    tmp31 = tmp29 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp13 - tmp23
    tmp36 = tmp35 * tmp34
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 * tmp38
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp12, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp34, None)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp43, rmask)
    tl.store(out_ptr2 + (x0), tmp23, None)
''')
