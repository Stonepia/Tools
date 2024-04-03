

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/vi/cvi4y7d4jwgi36txqcbhlubnrbsguvxh4kzcnim3cbbnmfrul2nv.py
# Source Nodes: [add_3, l__self___layer_0_attention_output_dense, l__self___layer_0_attention_output_dropout, l__self___layer_0_attention_output_layer_norm], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm]
# add_3 => add_8
# l__self___layer_0_attention_output_dense => add_7
# l__self___layer_0_attention_output_dropout => gt_1, mul_3, mul_4
# l__self___layer_0_attention_output_layer_norm => add_10, add_9, convert_element_type_7, convert_element_type_8, mul_5, mul_6, rsqrt, sub_6, var_mean
triton_per_fused_add_native_dropout_native_layer_norm_26 = async_compile.triton('triton_per_fused_add_native_dropout_native_layer_norm_26', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*i1', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_26(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = 768.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp15 - tmp25
    tmp38 = tmp37 * tmp36
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 * tmp40
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 + tmp43
    tmp45 = tmp44.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp14, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp36, None)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp45, rmask)
    tl.store(out_ptr2 + (x0), tmp25, None)
''')
