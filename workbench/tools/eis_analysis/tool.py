"""Keep established EIS analysis alongside conductivity-driven generation."""
from __future__ import annotations

from workbench.core.tool_base import Tool
from workbench.tools.eis.tool import EisTool


class EisAnalysisTool(EisTool):
    """Preserve EIS's callable interface while grouping its independent pages."""

    subtitle = (
        "Validate and fit impedance spectra, or generate catalyst-layer EIS "
        "from ionic and electronic conductivities."
    )

    def subtools(self) -> list[Tool]:
        from workbench.tools.eis_generator.tool import EisGeneratorTool

        analysis = EisTool()
        analysis.name = "Analysis"
        return [analysis, EisGeneratorTool()]
