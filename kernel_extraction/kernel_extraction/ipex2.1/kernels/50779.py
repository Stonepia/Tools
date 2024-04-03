

# Original file: ./nanogpt___60.0/nanogpt___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ay/cayvzqlyxohbz75k6x2mu7hjtf7nm57zdhechgaf3xpkkhhz3nft.py
# Source Nodes: [add_45, add_48, getitem_36, l__mod___transformer_h_11_attn_resid_dropout, l__mod___transformer_h_11_mlp_dropout, layer_norm_24], Original ATen: [aten.add, aten.clone, aten.index, aten.native_layer_norm, aten.slice]
# add_45 => add_91
# add_48 => add_96
# getitem_36 => index, slice_1, slice_2
# l__mod___transformer_h_11_attn_resid_dropout => clone_23
# l__mod___transformer_h_11_mlp_dropout => clone_24
# layer_norm_24 => add_97, add_98, convert_element_type_48, convert_element_type_49, mul_96, mul_97, rsqrt_24, sub_24, var_mean_24
triton_poi_fused_add_clone_index_native_layer_norm_slice_12 = async_compile.triton('triton_poi_fused_add_clone_index_native_layer_norm_slice_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_index_native_layer_norm_slice_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_index_native_layer_norm_slice_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (48384 + x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (48384 + x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (48384 + x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (63))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr4 + (63))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp17 = tl.load(in_ptr5 + (x0), xmask).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp11 = 768.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp8 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''')
