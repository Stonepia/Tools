

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/as/casxk2osp36jkng6mzu6gknmqro7ql3ypku7d7bucbmhxsqnm23x.py
# Source Nodes: [add_183, add_185, add_187, add_189, add_190, l__mod___model_decoder_block_13_layer_0_dropout, l__mod___model_decoder_block_13_layer_1_dropout, l__mod___model_decoder_block_13_layer__1__dropout, l__mod___model_decoder_block_14_layer_0_dropout, mean_92, mul_191, mul_192, pow_93, rsqrt_92, to_191, to_192], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_183 => add_235
# add_185 => add_238
# add_187 => add_240
# add_189 => add_243
# add_190 => add_244
# l__mod___model_decoder_block_13_layer_0_dropout => clone_229
# l__mod___model_decoder_block_13_layer_1_dropout => clone_232
# l__mod___model_decoder_block_13_layer__1__dropout => clone_234
# l__mod___model_decoder_block_14_layer_0_dropout => clone_237
# mean_92 => mean_92
# mul_191 => mul_191
# mul_192 => mul_192
# pow_93 => pow_93
# rsqrt_92 => rsqrt_92
# to_191 => convert_element_type_299
# to_192 => convert_element_type_300
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 1024.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp15 * tmp22
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp23, rmask & xmask)
''')
