

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/yg/cygvhtltzmhwmql6xxsbwi7izvqkuntfpek4vbefuxpspvcvdnhl.py
# Source Nodes: [l__mod___encoder_b_encoder], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# l__mod___encoder_b_encoder => add_92, add_93, add_94, clone_75, clone_76, convert_element_type_58, convert_element_type_59, mul_78, mul_79, rsqrt_27, sub_27, var_mean_27
triton_per_fused_add_clone_native_layer_norm_21 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_21', '''
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2464
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    r2 = rindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + (77*x0)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + (512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r2 + (512*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 49408, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 49408)) | ~xmask, "index out of bounds: 0 <= tmp1 < 49408")
    tmp2 = tl.load(in_ptr1 + (r2 + (512*tmp1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.full([1], 512, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tmp7 - tmp17
    tmp25 = 512.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp37, rmask & xmask)
''')