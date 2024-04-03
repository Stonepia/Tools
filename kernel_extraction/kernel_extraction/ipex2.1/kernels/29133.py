

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/zr/czrs35gjpgeqixqn22rv5g35baqwnjeibilvgvoaw5se3fk4odob.py
# Source Nodes: [add_18, add_19, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___mlp_drop2, getattr_getattr_l__mod___stages___4_____1___norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_18 => add_114
# add_19 => add_117
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_proj_drop => clone_70
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___mlp_drop2 => clone_72
# getattr_getattr_l__mod___stages___4_____1___norm => var_mean_20
triton_per_fused_add_clone_native_layer_norm_45 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_45', '''
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
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_45', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_45(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (240*x0)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr1 + (x0), tmp20, None)
''')
