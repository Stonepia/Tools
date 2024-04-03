

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/fx/cfxb27mhgzx7o52egt2lrx6xxtsa3ey3sx5k767vt2r3bqfk2sia.py
# Source Nodes: [setitem_10], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_10 => copy_10, slice_scatter_36, slice_scatter_37, slice_scatter_38
triton_poi_fused_copy_slice_scatter_15 = async_compile.triton('triton_poi_fused_copy_slice_scatter_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp37 = tl.load(in_ptr1 + (x1 + (513*(y0 % 256))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr4 + (y0 + (4096*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp4 = tl.full([1, 1], 257, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (x1 + (257*y0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.where(tmp6, tmp7, 0.0)
    tmp9 = tl.broadcast_to((y0 // 256), [XBLOCK, YBLOCK])
    tmp10 = tl.full([1, 1], 0, tl.int32)
    tmp11 = tmp9 == tmp10
    tmp12 = tl.load(in_ptr1 + (x1 + (513*(y0 % 256))), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.full([1, 1], 1, tl.int64)
    tmp14 = tmp9 >= tmp13
    tmp15 = tmp14 & tmp2
    tmp16 = tmp3 < tmp1
    tmp17 = tmp16 & tmp15
    tmp18 = (((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513
    tmp19 = tl.full([1, 1], 512, tl.int64)
    tmp20 = tmp18 < tmp19
    tmp21 = tmp20 & tmp17
    tmp22 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 15)) + ((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513)), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr3 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 15)) + ((x1 + (513*(y0 % 256))) % 512)), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.where(tmp21, tmp24, 0.0)
    tmp26 = tl.where(tmp17, tmp25, 0.0)
    tmp27 = tl.load(in_ptr4 + (y0 + (4096*x1)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.where(tmp16, tmp26, tmp27)
    tmp29 = tl.where(tmp15, tmp28, 0.0)
    tmp30 = tl.load(in_ptr4 + (y0 + (4096*x1)), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp14, tmp29, tmp30)
    tmp32 = tl.where(tmp11, tmp12, tmp31)
    tmp33 = tl.where(tmp5, tmp8, tmp32)
    tmp34 = tl.where(tmp2, tmp33, 0.0)
    tmp35 = (y0 // 256)
    tmp36 = tmp35 == tmp10
    tmp38 = tmp35 >= tmp13
    tmp39 = tmp16 & tmp38
    tmp40 = tmp20 & tmp39
    tmp41 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 15)) + ((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513)), tmp40 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr3 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 15)) + ((x1 + (513*(y0 % 256))) % 512)), tmp40 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tmp41 * tmp42
    tmp44 = tl.where(tmp40, tmp43, 0.0)
    tmp45 = tl.where(tmp39, tmp44, 0.0)
    tmp46 = tl.load(in_ptr4 + (y0 + (4096*x1)), tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tl.where(tmp16, tmp45, tmp46)
    tmp48 = tl.where(tmp38, tmp47, 0.0)
    tmp50 = tl.where(tmp38, tmp48, tmp49)
    tmp51 = tl.where(tmp36, tmp37, tmp50)
    tmp52 = tl.where(tmp2, tmp34, tmp51)
    tl.store(out_ptr0 + (x1 + (513*y0)), tmp52, xmask)
''')
