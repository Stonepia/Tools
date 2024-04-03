

# Original file: ./nanogpt___60.0/nanogpt___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/jj/cjjdtvuzayjwsqse7ahxfj3pxag6nymojahgdnwp55x7amdaodjm.py
# Source Nodes: [add_45, add_48, getitem_36, l__mod___transformer_h_11_attn_resid_dropout, l__mod___transformer_h_11_mlp_dropout, layer_norm_24], Original ATen: [aten.add, aten.clone, aten.index, aten.native_layer_norm, aten.slice]
# add_45 => add_103
# add_48 => add_108
# getitem_36 => index, slice_1, slice_2
# l__mod___transformer_h_11_attn_resid_dropout => clone_35
# l__mod___transformer_h_11_mlp_dropout => clone_36
# layer_norm_24 => add_109, add_110, mul_120, mul_121, rsqrt_24, sub_48, var_mean_24
triton_poi_fused_add_clone_index_native_layer_norm_slice_14 = async_compile.triton('triton_poi_fused_add_clone_index_native_layer_norm_slice_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_index_native_layer_norm_slice_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_index_native_layer_norm_slice_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (48384 + x0), xmask)
    tmp1 = tl.load(in_ptr1 + (48384 + x0), xmask)
    tmp3 = tl.load(in_ptr2 + (48384 + x0), xmask)
    tmp5 = tl.load(in_ptr3 + (63))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr4 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp16 = tl.load(in_ptr5 + (x0), xmask)
    tmp18 = tl.load(in_ptr6 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp4 - tmp6
    tmp10 = 768.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp7 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''')
