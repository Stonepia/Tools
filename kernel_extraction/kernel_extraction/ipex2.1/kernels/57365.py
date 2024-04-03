

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ye/cye6pe5nsiz4hujv5tvb3uazwntzyrke3ly2tvxqfcfos3qby3mx.py
# Source Nodes: [add_10, add_13, add_14, add_15, add_4, add_7, l__mod___transformer_blocks_0_input_sublayer_dropout, l__mod___transformer_blocks_0_output_sublayer_dropout, l__mod___transformer_blocks_1_input_sublayer_dropout, l__mod___transformer_blocks_1_output_sublayer_dropout, mean_4, mul_4, std_4, sub_4, truediv_6], Original ATen: [aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_10 => add_11
# add_13 => add_15
# add_14 => add_16
# add_15 => add_17
# add_4 => add_4
# add_7 => add_8
# l__mod___transformer_blocks_0_input_sublayer_dropout => clone_6
# l__mod___transformer_blocks_0_output_sublayer_dropout => clone_8
# l__mod___transformer_blocks_1_input_sublayer_dropout => clone_15
# l__mod___transformer_blocks_1_output_sublayer_dropout => clone_17
# mean_4 => mean_4
# mul_4 => mul_10
# std_4 => convert_element_type_16, convert_element_type_17, sqrt_4, var_4
# sub_4 => sub_6
# truediv_6 => div_8
triton_per_fused_add_clone_div_mean_mul_std_sub_10 = async_compile.triton('triton_per_fused_add_clone_div_mean_mul_std_sub_10', '''
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_div_mean_mul_std_sub_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_div_mean_mul_std_sub_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 768, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp10 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp13 / tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp8 - tmp31
    tmp33 = tmp28 * tmp32
    tmp34 = 767.0
    tmp35 = tmp27 / tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tmp33 / tmp39
    tmp42 = tmp40 + tmp41
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp42, rmask)
''')
