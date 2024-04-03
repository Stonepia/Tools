

# Original file: ./hf_T5_base___60.0/hf_T5_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/jm/cjmnu6jptm7smvsedi2doulbi7mihve4tjmv7fznjoknpn4jw5zv.py
# Source Nodes: [add_50, add_51, l__self___model_decoder_block_0_layer_1_enc_dec_attention_k, l__self___model_encoder_block_11_layer__1__dropout, mean_24, mul_51, mul_52, pow_25, rsqrt_24], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_50 => add_63
# add_51 => add_64
# l__self___model_decoder_block_0_layer_1_enc_dec_attention_k => convert_element_type_199
# l__self___model_encoder_block_11_layer__1__dropout => clone_60
# mean_24 => mean_24
# mul_51 => mul_51
# mul_52 => mul_52
# pow_25 => pow_25
# rsqrt_24 => rsqrt_24
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = 768.0
    tmp11 = tmp8 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp3 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp16, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp17, rmask)
''')
