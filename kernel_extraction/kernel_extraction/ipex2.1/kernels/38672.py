

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/il/cilbee6637hpu47hdai5sktwljubaqljchtiqy2xrfoy3i6nfws6.py
# Source Nodes: [abs_1, abs_2, abs_3, abs_4, add_14, add_17, add_20, add_21, add_22, add_23, add_24, add_25, add_26, grid_sample_4, grid_sample_5, l1_loss_fn, l1_loss_fn_1, l1_loss_fn_2, l1_loss_fn_3, l1_loss_fn_4, mean, mean_1, mean_2, mean_3, mse_loss_fn, mul_25, mul_26, mul_27, sub_17, sub_18, sub_19, sub_20], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.grid_sampler_2d, aten.mean, aten.mse_loss, aten.mul, aten.sub]
# abs_1 => abs_6
# abs_2 => abs_7
# abs_3 => abs_8
# abs_4 => abs_9
# add_14 => add_114
# add_17 => add_124
# add_20 => add_134
# add_21 => add_135
# add_22 => add_136
# add_23 => add_137
# add_24 => add_138
# add_25 => add_139
# add_26 => add_140
# grid_sample_4 => add_123, index_27, mul_222
# grid_sample_5 => add_133, index_31, mul_234
# l1_loss_fn => abs_1, mean, sub_109
# l1_loss_fn_1 => abs_2, mean_2, sub_111
# l1_loss_fn_2 => abs_3, mean_3, sub_112
# l1_loss_fn_3 => abs_4, mean_4, sub_123
# l1_loss_fn_4 => abs_5, mean_5, sub_134
# mean => mean_6
# mean_1 => mean_7
# mean_2 => mean_8
# mean_3 => mean_9
# mse_loss_fn => convert_element_type_354, convert_element_type_355, mean_1, pow_1, sub_110
# mul_25 => mul_235
# mul_26 => mul_236
# mul_27 => mul_237
# sub_17 => sub_135
# sub_18 => sub_136
# sub_19 => sub_137
# sub_20 => sub_138
triton_per_fused__to_copy_abs_add_grid_sampler_2d_mean_mse_loss_mul_sub_63 = async_compile.triton('triton_per_fused__to_copy_abs_add_grid_sampler_2d_mean_mse_loss_mul_sub_63', '''
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
    size_hints=[1, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_abs_add_grid_sampler_2d_mean_mse_loss_mul_sub_63', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_abs_add_grid_sampler_2d_mean_mse_loss_mul_sub_63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 181
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r0), rmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp20 = tl.load(in_out_ptr0 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, 1])
    tmp26 = tl.load(in_ptr4 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, 1])
    tmp29 = tl.load(in_ptr5 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp33 = tl.load(in_ptr6 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp37 = tl.load(in_ptr7 + (0))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, 1])
    tmp44 = tl.load(in_ptr8 + (0))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp22 = 2230272.0
    tmp23 = tmp21 / tmp22
    tmp24 = 204.0
    tmp25 = tmp23 * tmp24
    tmp28 = tmp27 / tmp22
    tmp31 = tmp30 / tmp22
    tmp32 = tmp28 + tmp31
    tmp35 = tmp34 / tmp22
    tmp36 = tmp32 + tmp35
    tmp39 = tmp38 / tmp22
    tmp40 = tmp36 + tmp39
    tmp41 = 102.0
    tmp42 = tmp40 * tmp41
    tmp43 = tmp25 + tmp42
    tmp46 = 5947392.0
    tmp47 = tmp45 / tmp46
    tmp48 = 0.005
    tmp49 = tmp47 * tmp48
    tmp50 = tmp43 + tmp49
    tmp51 = 1482624.0
    tmp52 = tmp4 / tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp9 / tmp51
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp53 + tmp55
    tmp57 = tmp14 / tmp51
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp19 / tmp51
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp58 + tmp60
    tmp62 = tmp56 + tmp61
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp50 + tmp63
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp64, None)
''')
