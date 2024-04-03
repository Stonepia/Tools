

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/xz/cxz3z555pcz6c7glpknze246cfbd5kzxpeq42pstxjphovj7tsgd.py
# Source Nodes: [add_4, add_5, add_6, l__mod___transformer_blocks_0_input_sublayer_dropout, mean_1, mul_1, std_1, sub_1, truediv_2], Original ATen: [aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_4 => add_4
# add_5 => add_5
# add_6 => add_6
# l__mod___transformer_blocks_0_input_sublayer_dropout => clone_6
# mean_1 => mean_1
# mul_1 => mul_1
# std_1 => convert_element_type_4, convert_element_type_5, sqrt_1, var_1
# sub_1 => sub_2
# truediv_2 => div_3
triton_per_fused_add_clone_div_mean_mul_std_sub_6 = async_compile.triton('triton_per_fused_add_clone_div_mean_mul_std_sub_6', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_div_mean_mul_std_sub_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_div_mean_mul_std_sub_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp4 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = 768.0
    tmp24 = tmp7 / tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp2 - tmp25
    tmp27 = tmp22 * tmp26
    tmp28 = 767.0
    tmp29 = tmp21 / tmp28
    tmp30 = tl.sqrt(tmp29)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tmp27 / tmp33
    tmp36 = tmp34 + tmp35
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp36, rmask)
''')
