

# Original file: ./hf_T5_generate__24_inference_64.4/hf_T5_generate__24_inference_64.4_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/eq/ceqkaunaoaqtsedt2ef53np35iety45svjvev5wtrp24anwb2qll.py
# Source Nodes: [add_47, add_50, add_52, add_53, l__self___decoder_block_5_layer_0_dropout, l__self___decoder_block_5_layer_1_dropout, l__self___decoder_block_5_layer__1__dropout, mean_18, mul_40, mul_41, mul_42, pow_19, rsqrt_18, to_40, to_41], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_47 => add_51
# add_50 => add_54
# add_52 => add_56
# add_53 => add_57
# l__self___decoder_block_5_layer_0_dropout => clone_32
# l__self___decoder_block_5_layer_1_dropout => clone_34
# l__self___decoder_block_5_layer__1__dropout => clone_36
# mean_18 => mean_18
# mul_40 => mul_40
# mul_41 => mul_41
# mul_42 => mul_42
# pow_19 => pow_19
# rsqrt_18 => rsqrt_18
# to_40 => convert_element_type_65
# to_41 => convert_element_type_66
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_17 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_17', '''
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
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp7 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp13 * tmp20
    tmp22 = 0.04419417382415922
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp23, rmask)
''')
