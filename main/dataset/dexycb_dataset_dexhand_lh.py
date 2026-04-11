from .decorators import register_manipdata
from .dexycb_dataset_dexhand_rh import DexYCBDatasetDexHandRH


@register_manipdata("dexycb_lh")
class DexYCBDatasetDexHandLH(DexYCBDatasetDexHandRH):
    pass
