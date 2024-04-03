

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/c2/cc25jkwfmafocg3t3hunxgmr7sv6wr37bq3a5gqgzbexsxh3uscl.py
# Source Nodes: [and__1, clone, ge_1, iadd_51, setitem_82, setitem_83, setitem_84, setitem_86, setitem_91, where_2], Original ATen: [aten.bitwise_and, aten.clone, aten.copy, aten.ge, aten.select_scatter, aten.slice_scatter, aten.where]
# and__1 => bitwise_and_1
# clone => clone
# ge_1 => ge_1
# iadd_51 => slice_scatter_39
# setitem_82 => copy_82, select_scatter_128
# setitem_83 => copy_83, select_scatter_129
# setitem_84 => copy_84, select_scatter_130, slice_scatter_20, slice_scatter_21, slice_scatter_22
# setitem_86 => slice_scatter_26
# setitem_91 => slice_scatter_42
# where_2 => where_2
triton_poi_fused_bitwise_and_clone_copy_ge_select_scatter_slice_scatter_where_68 = async_compile.triton('triton_poi_fused_bitwise_and_clone_copy_ge_select_scatter_slice_scatter_where_68', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_clone_copy_ge_select_scatter_slice_scatter_where_68', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bitwise_and_clone_copy_ge_select_scatter_slice_scatter_where_68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 15912)
    x4 = xindex
    x2 = (xindex // 78) % 204
    x0 = xindex % 3
    x5 = (xindex // 3) % 5304
    tmp27 = tl.load(in_ptr4 + (x4), xmask).to(tl.float32)
    tmp0 = x3
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-31824) + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.load(in_ptr1 + ((-31824) + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = tl.load(in_ptr2 + ((-31824) + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp5, tmp10, 0.0)
    tmp12 = x2
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp15 & tmp5
    tmp17 = x0
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp17 == tmp18
    tmp20 = tl.load(in_ptr3 + ((-10452) + x5 + (5200*x3)), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (x4), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.where(tmp16, tmp22, 0.0)
    tmp24 = tl.load(in_ptr4 + (x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.where(tmp15, tmp23, tmp24)
    tmp26 = tl.where(tmp5, tmp25, 0.0)
    tmp28 = tl.where(tmp5, tmp26, tmp27)
    tmp29 = tl.where(tmp5, tmp11, tmp28)
    tmp30 = tl.where(tmp5, tmp9, tmp29)
    tmp31 = tl.where(tmp5, tmp7, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, xmask)
''')
