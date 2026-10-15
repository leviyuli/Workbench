"""Group established LSV tools with integrated-reference cell analysis."""
from __future__ import annotations

from workbench.core.tool_base import Tool
from workbench.tools.lsv.tool import LsvTool


class LsvAnalysisTool(LsvTool):
    """Retain Decoupling's callable interface while grouping its UI pages."""

    name = "LSV analysis"
    subtitle = "Decouple, generate or compare electrode and full-cell LSV curves."

    def subtools(self) -> list[Tool]:
        from workbench.tools.integrated_re.tool import IntegratedReTool
        from workbench.tools.lsv_generator.tool import LsvGeneratorTool

        decoupling = LsvTool()
        decoupling.name = "Decoupling"
        generator = LsvGeneratorTool()
        generator.name = "Generator"
        return [decoupling, generator, IntegratedReTool()]
