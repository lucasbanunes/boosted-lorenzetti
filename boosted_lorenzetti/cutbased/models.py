from pathlib import Path
from typing import Annotated, Any, ClassVar, Mapping
from pydantic import BaseModel, ConfigDict, Field, computed_field
import pandas as pd
import numpy as np
import pandera.pandas as pa


class L1Info(BaseModel):

    VALID_L1_ETA_MAX: ClassVar[float] = 2.5

    eta_col: Annotated[
        str,
        Field(
            description="The name of the L1 eta coordinates column."
        )
    ] = 'trig_L1_eta'
    phi_col: Annotated[
        str,
        Field(
            description="The name of the L1 phi coordinates column."
        )
    ] = 'trig_L1_phi'


class L2Info(BaseModel):

    MAX_ABS_ETA: ClassVar[float] = 2.47

    e2tsts1_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster e2tsts1 column."
        )
    ] = 'trig_L2_calo_e2tsts1'
    emaxs1_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster emaxs1 column."
        )
    ] = 'trig_L2_calo_emaxs1'
    eta_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster eta coordinates column."
        )
    ] = 'trig_L2_calo_eta'
    phi_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster phi coordinates column."
        )
    ] = 'trig_L2_calo_phi'
    e277_col: Annotated[
        str,
        Field(
            description='The name of the L2 cluster e277 column.'
        )
    ] = 'trig_L2_calo_e277'
    e237_col: Annotated[
        str,
        Field(
            description='The name of the L2 cluster e237 column.'
        )
    ] = 'trig_L2_calo_e237'
    et_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster transverse energy column."
        )
    ] = 'trig_L2_calo_et'
    energy_sample_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster energy sample column."
        )
    ] = 'trig_L2_calo_energySample'
    weta2_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster weta2 column."
        )
    ] = 'trig_L2_calo_weta2'
    wstot_col: Annotated[
        str,
        Field(
            description="The name of the L2 cluster wstot column."
        )
    ] = 'trig_L2_calo_wstot'


CUT_MAP_DEFAULT_COLS = {
    'eratio': 999.0,
    'et': 0.0,
    'et2': 90000.0,
    'f1': 0.005,
    'f3': 99999.0,
    'had_et': 999.0,
    'had_et2': 999.0,
    'weta2': 99999.0,
    'wstot': 99999.0,
}

CUT_MAP_DF_SCHEMA_DICT = {
    'et_min': pa.Column(pa.Float, nullable=False),
    'et_max': pa.Column(pa.Float, nullable=False),
    'eta_min': pa.Column(pa.Float, nullable=False),
    'eta_max': pa.Column(pa.Float, nullable=False),
    'had_et': pa.Column(pa.Float, nullable=False),
    'eratio': pa.Column(pa.Float, nullable=False),
    'rcore': pa.Column(pa.Float, nullable=False),
}
for col in CUT_MAP_DEFAULT_COLS.keys():
    CUT_MAP_DF_SCHEMA_DICT[col] = pa.Column(pa.Float,
                                            nullable=False,
                                            required=False)


class ElectronCutMap(BaseModel):

    DEFAULT_COLS: ClassVar[dict[str, float]] = CUT_MAP_DEFAULT_COLS
    DF_SCHEMA: ClassVar[pa.DataFrameSchema] = pa.DataFrameSchema(
        CUT_MAP_DF_SCHEMA_DICT)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deta: Annotated[
        float,
        Field(
            description="The delta eta threshold between L1 and L2 objects.",
            ge=0.0
        )
    ] = 0.1
    dphi: Annotated[
        float,
        Field(
            description="The delta phi threshold between L1 and L2 objects.",
            ge=0.0
        )
    ] = 0.1
    cuts: Annotated[
        Path | str | pd.DataFrame,
        Field(
            description='Dataframe with the cut value. Can be a csv file path or the name of one of the predefined cutmaps in the "cutmaps" dir.',
            repr=False
        )
    ]

    def read_csv(self, filepath: Path) -> pd.DataFrame:
        return pd.read_csv(filepath, dtype=float)

    def model_post_init(self, context):

        if isinstance(self.cuts, str):
            cutmapdir = Path(__file__).parent / 'cutmaps'
            self.cuts = self.read_csv(cutmapdir / f'electron_{self.cuts}.csv')

        if isinstance(self.cuts, Path):
            self.cuts = self.read_csv(self.cuts)

        self.DF_SCHEMA.validate(self.cuts)

        self.cuts['et_interval'] = self.cuts.apply(
            lambda row: pd.Interval(
                left=row['et_min'], right=row['et_max'], closed='left'),
            axis=1
        )
        self.cuts['eta_interval'] = self.cuts.apply(
            lambda row: pd.Interval(
                left=row['eta_min'], right=row['eta_max'], closed='left'),
            axis=1
        )
        self.cuts = self.cuts.drop(
            ['et_min', 'et_max', 'eta_min', 'eta_max'],
            axis=1
        ).set_index(
            ['et_interval', 'eta_interval']
        )

        for col, default_val in self.DEFAULT_COLS.items():
            if col not in self.cuts.columns:
                self.cuts[col] = default_val

    @classmethod
    def from_yaml(cls, path: str) -> "ElectronCutMap":
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_sample_map(self, et: float, abs_eta: float) -> pd.Series:
        return self.cuts.loc[(et, abs_eta)]

    @computed_field
    @property
    def max_eta_bin(self) -> np.floating:
        return self.cuts.index.get_level_values(1).right.max()


VALID_PID_NAMES = (
    'vloose',
    'loose',
    'lhvloose',
    'lhloose',
    'medium',
    'lhmedium',
    'tight',
    'lhtight',
    'mergedtight'
)


class CaloSampling(object):
    # LAr barrel
    PreSamplerB = 0
    EMB1 = 1
    EMB2 = 2
    EMB3 = 3
    # LAr EM endcap
    PreSamplerE = 4
    EME1 = 5
    EME2 = 6
    EME3 = 7
    # Hadronic endcap
    HEC0 = 8
    HEC1 = 9
    HEC2 = 10
    HEC3 = 11
    # Tile barrel
    TileBar0 = 12
    TileBar1 = 13
    TileBar2 = 14
    # Tile gap (ITC & scint)
    TileGap1 = 15
    TileGap2 = 16
    TileGap3 = 17
    # Tile extended barrel
    TileExt0 = 18
    TileExt1 = 19
    TileExt2 = 20
    # Forward EM endcap
    FCAL0 = 21
    FCAL1 = 22
    FCAL2 = 23
    # MiniFCAL
    MINIFCAL0 = 24
    MINIFCAL1 = 25
    MINIFCAL2 = 26
    MINIFCAL3 = 27


class ElectronCutBasedModel(BaseModel):

    l1: Annotated[
        L1Info,
        Field(
            description="The L1 column information."
        )
    ] = L1Info()
    l2: Annotated[
        L2Info,
        Field(
            description="The L2 column information."
        )
    ] = L2Info()
    cut_map: Annotated[
        ElectronCutMap,
        Field(
            description="The cut map to be applied."
        )
    ]
    classification_col_name: Annotated[
        str,
        Field(
            description="The name of the output classification column."
        )
    ] = 'cutbased'
    reason_col_name: Annotated[
        str,
        Field(
            description="The name of the output reason column."
        )
    ] = 'cutbased_reason'

    def get_f1(self, data: Mapping[str, float]) -> float:
        energy_sample = data[self.l2.energy_sample_col]
        if abs(sum(energy_sample)) > 0.00001:
            f1 = (energy_sample[CaloSampling.EMB1] +
                  energy_sample[CaloSampling.EME1]) / sum(energy_sample)
        else:
            f1 = -1

        return f1

    def get_f3(self, data: Mapping[str, float]) -> float:
        energy_sample = data[self.l2.energy_sample_col]
        e0 = energy_sample[CaloSampling.PreSamplerB] + \
            energy_sample[CaloSampling.PreSamplerE]
        e1 = energy_sample[CaloSampling.EMB1] + \
            energy_sample[CaloSampling.EME1]
        e2 = energy_sample[CaloSampling.EMB2] + \
            energy_sample[CaloSampling.EME2]
        e3 = energy_sample[CaloSampling.EMB3] + \
            energy_sample[CaloSampling.EME3]
        eallsamples = float(e0 + e1 + e2 + e3)
        f3 = e3 / eallsamples if abs(eallsamples) > 0. else 0.
        return f3

    def get_eratio(self, data: Mapping[str, float]) -> float:
        eratio_denominator = data[self.l2.emaxs1_col] + \
            data[self.l2.e2tsts1_col]
        eratio_numerator = data[self.l2.emaxs1_col] - data[self.l2.e2tsts1_col]
        eratio = eratio_numerator / eratio_denominator if eratio_denominator != 0 else -1
        return eratio

    def apply_cut(self, data: Mapping[str, float], return_reason: bool = False) -> dict[str, Any]:
        result = {
            self.classification_col_name: -1,
        }
        if abs(data[self.l1.eta_col]) > self.l1.VALID_L1_ETA_MAX:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'invalid_l1_eta'
            return result

        if abs(data[self.l2.eta_col] - data[self.l1.eta_col]) > self.cut_map.deta:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'deta_too_big'
            return result

        l1_phi = data[self.l1.phi_col]
        if abs(l1_phi) > np.pi:
            l1_phi = l1_phi - (2 * np.pi)
        dphi = abs(data[self.l2.phi_col] - l1_phi)
        if dphi > np.pi:
            dphi = 2 * np.pi - dphi
        if dphi > self.cut_map.dphi:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'dphi_too_big'
            return result

        l2_abs_eta = abs(data[self.l2.eta_col])
        if l2_abs_eta > self.cut_map.max_eta_bin:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'invalid_l2_eta'
            return result

        sample_cut_map = self.cut_map.get_sample_map(
            data[self.l2.et_col],
            l2_abs_eta
        )

        if data[self.l2.e277_col] == 0:
            rcore = -1
        else:
            rcore = data[self.l2.e237_col] / data[self.l2.e277_col]
        if rcore < sample_cut_map.rcore:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'rcore_too_small'
            return result

        eratio = self.get_eratio(data)
        in_crack = l2_abs_eta > 2.37 or (1.37 < l2_abs_eta < 1.52)
        low_f1 = self.get_f1(data) < sample_cut_map.f1
        if not (in_crack or low_f1) and eratio < sample_cut_map.eratio:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'eratio_too_small'
            return result

        if in_crack:
            eratio = -1

        if data[self.l2.et_col]*1e-3 < sample_cut_map.et:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'et_too_small'
            return result

        if data[self.l2.et_col] > sample_cut_map.et2:
            hadet_thr = sample_cut_map.had_et
        else:
            hadet_thr = sample_cut_map.had_et2
        had_et = data[self.l2.et_col] - data[self.l2.e277_col]
        had_et = had_et / np.cosh(l2_abs_eta)
        if had_et > hadet_thr:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'had_et_too_big'
            return result

        if data[self.l2.weta2_col] > sample_cut_map.weta2:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'weta2_too_big'
            return result

        if data[self.l2.wstot_col] >= sample_cut_map.wstot:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'wstot_too_big'
            return result

        if self.get_f3(data) > sample_cut_map.f3:
            result[self.classification_col_name] = 0
            if return_reason:
                result[self.reason_col_name] = 'f3_too_big'
            return result

        result[self.classification_col_name] = 1
        if return_reason:
            result[self.reason_col_name] = ''

        return result

    def predict(self, data: pd.DataFrame, return_reason: bool = False) -> pd.DataFrame:
        results = data.apply(self.apply_cut, axis=1, result_type='expand', return_reason=return_reason)
        return results
