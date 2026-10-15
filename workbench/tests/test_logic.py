"""Logic-only tests (no Qt). Run with: pytest workbench/tests"""
import numpy as np
import pandas as pd

from workbench import theme
from workbench.core.io import write_xlsx
from workbench.tools.ink_calc.logic import calculate_ink_recipe
from workbench.tools.ecsa.logic_ecsa import compute_cdl_ecsa
from workbench.tools.lsv.logic_tafel import (
    calculate_RCL_overpotential,
    calculate_RCL_utilization,
    iterative_tafel_fit,
)
from workbench.tools.lsv.tool import (
    FARADAY,
    REFERENCE_OPTIONS,
    R_GAS,
    LsvTool,
    _plot_curves,
    convert_reference_to_rhe,
)
from workbench.tools.it_integrate.logic import integrate_potentiostatic_tests
from workbench.tools.eis.logic_eis import EisLogic
from workbench.tools.ftir.logic import (
    apply_vertical_offset,
    convert_t_to_a,
    find_spectrum_peaks,
    normalize_spectrum,
    read_ftir_file,
    read_xps_file,
)
from workbench.tools.ftir.tool import FtirTool


def test_ink_recipe_area_mode():
    params = {
        "mode": "area", "length": 5.0, "width": 5.0, "loading": 1.0,
        "ink_density": 2.0,
        "ionomers": [{"ratio": 20.0, "conc": 5.0, "density": 0.78}],
        "extras": [],
    }
    res = calculate_ink_recipe(params)
    assert res["catalyst_mg"] == 25.0                 # 1 * 5 * 5
    # m_i = 25 * 0.2 / 0.8 = 6.25
    assert abs(res["total_ionomer_mg"] - 6.25) < 1e-9
    assert res["ink_volume_ml"] == 12.5               # 25 / 2
    assert res["adjusted"] is not None                 # area mode -> loss computed


def test_ecsa_linear_recovery():
    # Construct synthetic CV cycles with a known C_dl.
    Cdl_true = 0.01  # F/cm²
    rows = []
    E = np.linspace(0.0, 0.2, 201)
    for rate_mV in (20, 40, 60, 80):
        v = rate_mV * 1e-3
        jdl = Cdl_true * v
        # anodic +jdl, cathodic -jdl across the window
        i = np.where(E <= 0.1, jdl, -jdl)
        for e, ii in zip(E, i):
            rows.append({"scan_rate": rate_mV, "potential": e, "current": ii})
    long_df = pd.DataFrame(rows)
    out = compute_cdl_ecsa(long_df, Cs_uF_per_cm2=40.0, n_nearest=5)
    assert abs(out["Cdl"] - Cdl_true) < 1e-3
    assert out["r2"] > 0.99
    assert out["ECSA"] is not None


def test_tafel_fit_runs_on_synthetic():
    # Synthetic Tafel: eta = b*log10(i/i0); build E = E_rev + eta
    E_rev, b, i0 = 1.23, 0.06, 1e-6
    i = np.logspace(-3, 0, 400)
    V = E_rev + b * np.log10(i / i0)
    out, err = iterative_tafel_fit(V, i, E_rev, HFR=0.0, R_CL=0.0,
                                   i_lower=0.005, i_upper=0.5)
    assert out is not None
    fit = out[0]
    assert abs(fit["b_kin"] - b) < 0.02
    details = fit["candidate_details"]
    eta = V - E_rev
    in_window = (eta >= details["window"][0]) & (eta <= details["window"][1])
    assert np.allclose(details["current_range"],
                       [np.min(i[in_window]), np.max(i[in_window])])


def test_rcl_utilization_is_the_value_used_for_rcl_overpotential():
    current = np.array([0.0, 0.1, 0.5, 1.0])
    tafel_slope = 0.06
    utilization = calculate_RCL_utilization(current, R_CL=0.2, b_kin=tafel_slope)
    overpotential = calculate_RCL_overpotential(current, R_CL=0.2,
                                                b_kin=tafel_slope)

    assert utilization[0] == 1.0
    assert np.all(np.diff(utilization) < 0)
    assert np.allclose(overpotential, -tafel_slope * np.log10(utilization))


def test_lsv_her_decoupling_end_to_end(tmp_path):
    # Synthetic HER LSV: negative current, V = -b*log10(|i|/i0) below E_rev = 0.
    b, i0 = 0.06, 1e-6
    i_mag = np.logspace(-3, 0, 400)        # 1 mA/cm² .. 1 A/cm²
    i = -i_mag                              # cathodic (negative) current
    V = -b * np.log10(i_mag / i0)           # native potential, all < 0
    csv = tmp_path / "her.csv"
    lines = ["E(V),j(A/cm2)"] + [f"{v},{ii}" for v, ii in zip(V, i)]
    csv.write_text("\n".join(lines), encoding="utf-8")

    params = {
        "electrode_setup": "Three-electrode",
        "reaction": "Auto", "reference": "RHE", "ph": 0.0, "temperature": 298.15,
        "hfr": 0.0, "rcl": 0.0, "i_lower": 0.02, "i_upper": 0.8,
        "target_i": 0.5, "x_log": False,
    }
    res = LsvTool().run([csv], params)

    # Auto-detected HER and recovered the Tafel slope magnitude.
    assert any("HER" in w for w in res.warnings)
    assert abs(res.scalars["b_kin (V/dec)"] - b) < 0.02
    assert "Fit current range (A/cm²)" in res.scalars
    fitting_info = res.tables["Fitting info"].set_index("Parameter")["Value"]
    assert "Fit current lower (A/cm²)" in fitting_info.index
    assert "Fit current upper (A/cm²)" in fitting_info.index

    # Native-sign display: original LSV and the kinetic curve sit below E_rev = 0.
    curves = res.tables["Curves (stacked)"]
    assert (curves["Original LSV (RHE)"] < 0).all()
    assert (curves["E_rev + η_kin"] <= curves["E_rev"] + 1e-9).all()

    # The exact RCL utilization is plotted and included as its own Excel sheet.
    utilization = res.tables["Catalyst utilization"]
    assert np.allclose(utilization["Current Density Magnitude (A/cm²)"], i_mag)
    assert np.allclose(utilization["Catalyst Utilization U_J (-)"], 1.0)
    utilization_line = res.figures["Catalyst utilization"].axes[0].lines[0]
    assert np.allclose(utilization_line.get_xdata(), i_mag)
    assert np.allclose(utilization_line.get_ydata(), 1.0)
    excel = next(a for a in LsvTool().exportable_artifacts(res) if a.kind == "xlsx")
    assert "Catalyst utilization" in excel.payload
    export_path = tmp_path / "lsv_results.xlsx"
    write_xlsx(excel.payload, export_path)
    with pd.ExcelFile(export_path) as workbook:
        assert "Catalyst utilization" in workbook.sheet_names


def test_lsv_reference_names_are_chemically_complete():
    assert list(REFERENCE_OPTIONS) == [
        "RHE",
        "SHE",
        "Ag/AgCl (sat. KCl, 25 °C)",
        "Ag/AgCl (3 M KCl, 25 °C)",
        "SCE (sat. KCl, 25 °C)",
        "Hg/HgO (1 M KOH, 25 °C)",
        "Custom E_ref vs SHE",
    ]


def test_lsv_reference_to_rhe_uses_temperature_dependent_nernst_term():
    temperature = 378.0
    ph = 14.0
    converted, shift, e_ref = convert_reference_to_rhe(
        np.array([0.0]),
        "Hg/HgO (1 M KOH, 25 °C)",
        ph,
        temperature,
    )
    expected = 0.1053 + (2.303 * R_GAS * temperature / FARADAY) * ph
    assert e_ref == 0.1053
    assert np.isclose(shift, expected)
    assert np.isclose(converted[0], expected)


def test_lsv_reference_to_rhe_supports_custom_reference_potential():
    custom, custom_shift, _ = convert_reference_to_rhe(
        [0.25], "Custom E_ref vs SHE", 13.0, 298.15,
        custom_e_ref_vs_she=0.1234,
    )
    assert custom[0] > 0.25
    assert custom_shift > 0.1234


def test_lsv_decoupled_plot_can_exchange_xy_axes():
    current = np.array([0.01, 0.1, 1.0])
    curves = [
        np.array([1.23, 1.23, 1.23]),
        np.array([1.30, 1.35, 1.40]),
        np.array([1.31, 1.37, 1.43]),
        np.array([1.32, 1.39, 1.46]),
        np.array([1.33, 1.41, 1.49]),
        np.array([1.34, 1.43, 1.52]),
    ]
    fig = _plot_curves(current, curves, swap_axes=True)
    ax = fig.axes[0]
    original_lsv = ax.lines[-1]

    assert "Potential" in ax.get_xlabel()
    assert "Current density" in ax.get_ylabel()
    assert np.allclose(original_lsv.get_xdata(), curves[-1])
    assert np.allclose(original_lsv.get_ydata(), current)


def test_lsv_decoupled_plot_moves_log_current_scale_when_axes_are_exchanged():
    current = np.array([-0.01, -0.1, -1.0])
    curves = [np.array([0.0, -0.1, -0.2])] * 6
    fig = _plot_curves(current, curves, x_log=True, is_her=True, swap_axes=True)
    ax = fig.axes[0]

    assert ax.get_yscale() == "log"
    assert np.allclose(ax.lines[-1].get_ydata(), np.abs(current))


def test_it_integrate_potential_mode(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("T(s)\tE(V)\n0\t1.0\n1\t1.1\n2\t1.2\n", encoding="utf-8")
    f2.write_text("T(s)\tE(V)\n0\t2.0\n1\t2.1\n2\t2.2\n", encoding="utf-8")
    out = integrate_potentiostatic_tests([str(f1), str(f2)], interval_s=None, trace_type="E_t")
    assert out["headers"] == ["T(s)", "E(V)"]
    # Second file's time should be shifted to continue after the first.
    t = out["data"][:, 0]
    assert t[-1] > 2.0
    assert out["n_out"] == 6


def test_it_integrate_current_mode_without_potential_column(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("i(A/cm2)\tT(s)\n1.0\t0\n1.1\t1\n", encoding="utf-8")
    f2.write_text("i(A/cm2)\tT(s)\n2.0\t0\n2.1\t1\n", encoding="utf-8")
    out = integrate_potentiostatic_tests([str(f1), str(f2)], interval_s=None, trace_type="i_t")
    assert out["headers"] == ["i(A/cm2)", "T(s)"]
    assert out["time_header"] == "T(s)"
    assert np.allclose(out["data"][:, 1], [0.0, 1.0, 2.0, 3.0])
    assert out["n_out"] == 4


def test_ftir_header_detection_and_t_to_a(tmp_path):
    csv = tmp_path / "ftir_t.csv"
    csv.write_text("Wavenumber (cm-1),T%\n4000,100\n3000,10\n", encoding="utf-8")
    spec = read_ftir_file(csv)
    assert spec.source_type == "T"
    assert spec.x_column == "Wavenumber (cm-1)"
    assert spec.y_column == "T%"
    assert np.allclose(spec.wavenumber, [3000, 4000])
    assert np.allclose(convert_t_to_a(np.array([100.0, 10.0])), [0.0, 1.0])


def test_ftir_absorbance_header_detection(tmp_path):
    txt = tmp_path / "ftir_a.txt"
    txt.write_text("cm-1\tA%\n1000\t0.2\n900\t0.4\n", encoding="utf-8")
    spec = read_ftir_file(txt)
    assert spec.source_type == "A"
    assert np.allclose(spec.wavenumber, [900, 1000])
    assert np.allclose(spec.intensity, [0.4, 0.2])


def test_ftir_invalid_t_normalization_and_offset():
    converted = convert_t_to_a(np.array([100.0, 0.0, -5.0, 10.0]))
    assert np.allclose(converted[[0, 3]], [0.0, 1.0])
    assert np.isnan(converted[1]) and np.isnan(converted[2])
    assert np.allclose(
        normalize_spectrum(np.array([2.0, 4.0]), "Maximum normalization"),
        [0.5, 1.0],
    )
    assert np.allclose(
        normalize_spectrum(np.array([2.0, 4.0]), "Min-max normalization"),
        [0.0, 1.0],
    )
    assert np.allclose(apply_vertical_offset(np.array([1.0, 2.0]), 2, 0.5), [2.0, 3.0])


def test_spectrum_peak_detection_handles_maxima_and_transmittance_minima():
    x = np.linspace(500.0, 4000.0, 1001)
    peak_a = np.exp(-0.5 * ((x - 1200.0) / 25.0) ** 2)
    peak_b = 0.7 * np.exp(-0.5 * ((x - 2850.0) / 40.0) ** 2)
    absorbance = 0.1 + peak_a + peak_b
    transmittance = 95.0 - 30.0 * peak_a - 20.0 * peak_b

    maxima = find_spectrum_peaks(x, absorbance)
    minima = find_spectrum_peaks(x, transmittance, minima=True)

    maxima_x = np.asarray(maxima["x"])[np.asarray(maxima["indices"], dtype=int)]
    minima_x = np.asarray(minima["x"])[np.asarray(minima["indices"], dtype=int)]
    assert np.any(np.isclose(maxima_x, 1200.0, atol=5.0))
    assert np.any(np.isclose(maxima_x, 2850.0, atol=5.0))
    assert np.any(np.isclose(minima_x, 1200.0, atol=5.0))
    assert np.any(np.isclose(minima_x, 2850.0, atol=5.0))


def test_spectrum_peak_filters_reject_narrow_and_low_prominence_features():
    x = np.arange(401, dtype=float)
    broad = np.exp(-0.5 * ((x - 100.0) / 8.0) ** 2)
    weak = 0.08 * np.exp(-0.5 * ((x - 250.0) / 8.0) ** 2)
    spike = np.zeros_like(x)
    spike[330] = 0.7
    found = find_spectrum_peaks(
        x, broad + weak + spike,
        prominence_percent=10.0,
        noise_multiplier=0.0,
        minimum_width=3.0,
        smoothing_window=3,
    )
    peak_x = np.asarray(found["x"])[np.asarray(found["indices"], dtype=int)]
    assert np.any(np.isclose(peak_x, 100.0, atol=1.0))
    assert not np.any(np.isclose(peak_x, 250.0, atol=1.0))
    assert not np.any(np.isclose(peak_x, 330.0, atol=1.0))


def test_ftir_schema_uses_simplified_defaults():
    specs = FtirTool().parameter_schema()
    by_key = {s.key: s for s in specs}
    assert by_key["display_mode"].choices == ["T", "A"]
    assert by_key["display_mode"].default == "T"
    assert by_key["normalization"].default == "Min-max normalization"
    assert by_key["vertical_offset"].default == 1.0
    assert by_key["technique"].choices == ["FTIR", "XPS"]
    assert by_key["display_mode"].show_if == ("technique", "FTIR")
    assert by_key["custom_x_range"].default is False
    assert by_key["show_reference_line"].default is False
    assert by_key["reference_x"].default == 1300.0
    assert by_key["reference_x"].show_if == ("show_reference_line", True)
    assert not {
        "peak_file", "peak_prominence_percent", "peak_noise_multiplier",
        "peak_min_width", "peak_smoothing_window",
    } & by_key.keys()
    assert FtirTool().actions() == []
    assert "curve_names" not in by_key
    assert "line_width" not in by_key
    assert "legend_position" not in by_key


def test_shared_theme_uses_minimal_anthropic_palette():
    assert theme.ACCENT == "#D97757"
    assert theme.SERIES_COLORS[0] == theme.ACCENT
    assert theme.TEXT == "#181817"
    assert len(theme.SERIES_COLORS) == len(set(theme.SERIES_COLORS)) == 10
    stylesheet = theme.stylesheet()
    assert "{{" not in stylesheet
    assert f"background: {theme.TEXT};" in stylesheet


def test_ftir_tool_end_to_end_and_data_only_exports(tmp_path):
    t_file = tmp_path / "t.dat"
    a_file = tmp_path / "a.txt"
    t_file.write_text("cm-1 T%\n4000 100\n3000 10\n", encoding="utf-8")
    a_file.write_text("cm-1\tA%\n4000\t0.1\n3000\t0.3\n", encoding="utf-8")
    params = {
        "display_mode": "A",
        "normalization": "None",
        "vertical_offset": 0.2,
        "hide_y_ticks": False,
    }

    tool = FtirTool()
    result = tool.run([t_file, a_file], params)
    plot_data = result.tables["FTIR plot data"]
    assert list(plot_data.columns) == [
        "t Wavenumber (cm-1)", "t Y display",
        "a Wavenumber (cm-1)", "a Y display",
    ]
    assert plot_data.shape == (2, 4)
    assert np.allclose(plot_data["t Y display"], [1.0, 0.0])
    assert np.allclose(plot_data["a Y display"], [0.5, 0.3])

    artifacts = tool.exportable_artifacts(result)
    assert [a.kind for a in artifacts] == ["csv", "xlsx"]
    assert all(a.kind not in {"png", "pdf", "svg"} for a in artifacts)


def test_ftir_find_peaks_returns_peak_x_axis_page_and_exports(tmp_path):
    path = tmp_path / "ftir_peaks.csv"
    x = np.linspace(500.0, 4000.0, 1001)
    transmittance = 95.0 - 30.0 * np.exp(-0.5 * ((x - 1710.0) / 30.0) ** 2)
    lines = ["Wavenumber (cm-1),T%"] + [
        f"{xx},{yy}" for xx, yy in zip(x, transmittance)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    tool = FtirTool()
    result = tool.find_peaks([path], {"technique": "FTIR", "peak_file": str(path)})

    peaks = result.tables["Peaks"]
    assert list(peaks.columns) == [
        "Peak", "Peak wavenumber (cm-1)", "Intensity", "Prominence",
        "Peak width (points)", "Peak width (cm-1)",
    ]
    assert np.any(np.isclose(peaks["Peak wavenumber (cm-1)"], 1710.0, atol=5.0))
    assert result.scalars["Peak direction"] == "Minima"
    assert [artifact.default_filename for artifact in tool.exportable_artifacts(result)] == [
        "ftir_peaks.csv", "ftir_peaks.xlsx",
    ]


def test_xps_vertical_offset_custom_view_and_full_range_export(tmp_path):
    first = tmp_path / "survey.csv"
    second = tmp_path / "detail.dat"
    first.write_text(
        "Binding Energy (eV),Counts/s\n1200,10\n600,20\n0,30\n",
        encoding="utf-8",
    )
    second.write_text(
        "Binding Energy (eV)\tIntensity\n1200\t100\n600\t200\n0\t300\n",
        encoding="utf-8",
    )

    spec = read_xps_file(first)
    assert spec.technique == "XPS"
    assert spec.x_column == "Binding Energy (eV)"
    assert np.allclose(spec.x, [0, 600, 1200])

    result = FtirTool().run([first, second], {
        "technique": "XPS",
        "normalization": "Min-max normalization",
        "vertical_offset": 1.0,
        "hide_y_ticks": False,
        "custom_x_range": True,
        "x_display_min": 400.0,
        "x_display_max": 800.0,
        "show_reference_line": True,
        "reference_x": 600.0,
    })

    exported = result.tables["XPS plot data"]
    assert list(exported.columns) == [
        "survey Binding energy (eV)", "survey Y display",
        "detail Binding energy (eV)", "detail Y display",
    ]
    assert exported.shape == (3, 4)
    assert set(exported["survey Binding energy (eV)"]) == {0.0, 600.0, 1200.0}
    assert np.allclose(exported["detail Y display"], [2.0, 1.5, 1.0])

    ax = result.figures["XPS spectra"].axes[0]
    assert ax.get_xlabel() == "Binding energy (eV)"
    assert ax.get_ylabel() == "Intensity (normalized)"
    assert np.allclose(ax.get_xlim(), [800.0, 400.0])
    reference_line = ax.lines[-1]
    assert np.allclose(reference_line.get_xdata(), [600.0, 600.0])
    assert reference_line.get_color() == "gray"
    assert reference_line.get_linestyle() == ":"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "survey", "detail",
    ]
    assert result.scalars["Vertical reference line"] == "600 eV"

    artifacts = FtirTool().exportable_artifacts(result)
    assert [a.default_filename for a in artifacts] == [
        "xps_plot_data.csv", "xps_plot_data.xlsx",
    ]


def test_xps_background_is_plotted_normalized_offset_and_exported(tmp_path):
    path = tmp_path / "c1s.csv"
    path.write_text(
        "Binding Energy (eV),Counts,Background Counts\n"
        "280,20,12\n285,60,16\n290,100,20\n",
        encoding="utf-8",
    )

    spec = read_xps_file(path)
    assert spec.background_column == "Background Counts"
    assert np.allclose(spec.background, [12, 16, 20])

    result = FtirTool().run([path], {
        "technique": "XPS",
        "normalization": "Min-max normalization",
        "vertical_offset": 1.0,
    })

    exported = result.tables["XPS plot data"]
    assert list(exported.columns) == [
        "c1s Binding energy (eV)", "c1s Y display", "c1s Background display",
    ]
    assert np.allclose(exported["c1s Y display"], [0.0, 0.5, 1.0])
    assert np.allclose(exported["c1s Background display"], [-0.1, -0.05, 0.0])
    assert result.scalars["Background curves plotted"] == 1

    ax = result.figures["XPS spectra"].axes[0]
    assert len(ax.lines) == 2
    assert ax.lines[1].get_linestyle() == "--"
    assert ax.lines[0].get_color() == theme.SERIES_COLORS[0]
    assert ax.lines[1].get_color() == theme.SERIES_COLORS[0]
    assert np.allclose(ax.lines[1].get_ydata(), [-0.1, -0.05, 0.0])


def test_xps_avantage_clipboard_text_with_split_header(tmp_path):
    path = tmp_path / "n1s.txt"
    path.write_text(
        "Axis\tEnergy\tElements=\t3\n\n"
        "C:\\Temp\\N1s.vgd\n\n"
        "Binding Energy (E)\t\t\t\tBackgnd.\t\n"
        "eV\t\tCounts / s\t\tCounts / s\n"
        "411.180\t\t13123.0\t\t0.00000\t\n"
        "411.080\t\t13162.1\t\t13159.3\t\n"
        "410.980\t\t13243.7\t\t0.00000\t\n",
        encoding="utf-8",
    )

    spec = read_xps_file(path)
    assert spec.x_column == "Binding Energy (eV)"
    assert spec.y_column == "Counts / s"
    assert spec.background_column == "Background Counts / s"
    assert np.allclose(spec.x, [410.980, 411.080, 411.180])
    assert np.allclose(spec.intensity, [13243.7, 13162.1, 13123.0])
    assert np.allclose(spec.background, [0.0, 13159.3, 0.0])

    result = FtirTool().run([path], {
        "technique": "XPS",
        "normalization": "Min-max normalization",
        "vertical_offset": 0.0,
    })
    background = result.tables["XPS plot data"]["n1s Background display"]
    assert np.isnan(background.iloc[0])
    assert np.isfinite(background.iloc[1])
    assert np.isnan(background.iloc[2])
    assert any("ignored 2 zero background endpoint" in warning for warning in result.warnings)


def test_eis_two_rc_roundtrip():
    # Synthesize a faradaic TLM with TWO RC units (one distributed rung + one
    # series R||CPE arc) and check the fit recovers the planted resistances.
    logic = EisLogic()
    logic.fit_n_jobs = 1  # keep the test in-process (no worker-process spawning)
    model = EisLogic.FARADAIC_TML_MODEL
    f = np.logspace(5, -2, 140)
    # canonical: [L_wire, HFR, R_CL, Q_dl, Phi, Theta, R_k] + [R_2, Q_2, Phi_2]
    canon = [0.0, 0.5, 0.6, 0.10, 0.85, 1.0, 3.0, 0.5, 0.05, 0.90]
    z = logic.evaluate_model(model, canon, f)

    # init from the truth-ish values: [HFR, R_CL, Q_dl, Phi, R_k, R_2, Q_2, Phi_2]
    init = [0.5, 0.6, 0.10, 0.85, 3.0, 0.5, 0.05, 0.90]
    out, err = logic.fit_impedance(model, init, f, z, n_restarts=6, random_seed=0)
    assert err is None
    p, se, _resid, info = out
    assert len(p) == 10                      # 7 core + one extra triple
    assert abs(p[2] - 0.6) < 0.06            # R_CL
    assert abs(p[6] - 3.0) < 0.3             # R_k
    assert abs(p[7] - 0.5) < 0.05            # R_2 (series arc)


def test_eis_stable_coth_matches_analytic():
    # The overflow-safe coth/tanh must equal the analytic values and stay finite
    # where the old (e^x±e^-x) form overflowed to inf/nan at high ω (large |x|).
    logic = EisLogic()
    x = np.array([0.05 + 0.02j, 1.0 + 0.5j, 8.0 + 3.0j, 40.0 + 10.0j])
    assert np.allclose(logic.JPcoth(x), 1.0 / np.tanh(x))
    assert np.allclose(logic.JPtanh(x), np.tanh(x))
    assert np.all(np.isfinite(logic.JPcoth(np.array([300.0 + 300.0j]))))


def test_eis_faradaic_limits():
    # Two physics limits that anchor the faradaic model to Eikerling–Kornyshev.
    logic = EisLogic()
    f = np.logspace(5, -2, 120)
    hfr, rcl, qdl, phi, rk = 0.5, 0.6, 0.10, 0.85, 3.0

    # (1) R_k -> inf collapses the faradaic TML onto the blocking "Transmission Line".
    z_far = logic.evaluate_model(
        EisLogic.FARADAIC_TML_MODEL, [0.0, hfr, rcl, qdl, phi, 1.0, np.inf], f)
    z_block = logic.evaluate_model(
        "Transmission Line", [0.0, hfr, rcl, qdl, phi, 1.0, np.inf], f)
    assert np.allclose(z_far, z_block, rtol=1e-6, atol=1e-9)

    # (2) DC limit == HFR + sqrt(R_CL*R_k)*coth(sqrt(R_CL/R_k))  (E–K Eq. 11/26).
    z_lf = logic.evaluate_model(
        EisLogic.FARADAIC_TML_MODEL, [0.0, hfr, rcl, qdl, phi, 1.0, rk],
        np.array([1e-9]))[0]
    r_diff = hfr + np.sqrt(rcl * rk) / np.tanh(np.sqrt(rcl / rk))
    assert abs(np.real(z_lf) - r_diff) < 1e-3
    assert abs(np.imag(z_lf)) < 1e-3


def test_eis_seed_extra_arcs_locates_planted_arc():
    # Data-driven arc seeding returns finite, in-range guesses for the extra unit.
    from workbench.tools.eis.tool import _seed_extra_arcs
    logic = EisLogic()
    f = np.logspace(5, -2, 200)
    canon = [0.0, 0.5, 0.3, 0.10, 0.90, 1.0, 2.0, 0.5, 0.5, 0.90]  # + one series arc
    z = logic.evaluate_model(EisLogic.FARADAIC_TML_MODEL, canon, f)
    seeds = _seed_extra_arcs(f, z, n_rc=2)
    assert 2 in seeds
    r2, q2, phi2 = seeds[2]
    assert np.isfinite(r2) and r2 > 0
    assert 1e-9 <= q2 <= 1e3
    assert _seed_extra_arcs(f, z, n_rc=1) == {}   # no extras requested


def test_eis_modulus_weighting_runs():
    # Modulus weighting (the default) must still recover the planted parameters
    # on a clean spectrum.
    logic = EisLogic()
    logic.fit_n_jobs = 1
    model = EisLogic.FARADAIC_TML_MODEL
    f = np.logspace(5, -2, 140)
    canon = [0.0, 0.5, 0.6, 0.10, 0.85, 1.0, 3.0, 0.5, 0.05, 0.90]
    z = logic.evaluate_model(model, canon, f)
    init = [0.5, 0.6, 0.10, 0.85, 3.0, 0.5, 0.05, 0.90]
    out, err = logic.fit_impedance(model, init, f, z, n_restarts=4, random_seed=0,
                                   weighting="modulus")
    assert err is None
    p = out[0]
    assert abs(p[1] - 0.5) < 0.03 and abs(p[2] - 0.6) < 0.06   # HFR, R_CL
