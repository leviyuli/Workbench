"""Tool registry. Adding a tool = add its class here."""
from __future__ import annotations

from workbench.core.tool_base import Tool


def build_tools() -> list[Tool]:
    from workbench.tools.lsv_analysis.tool import LsvAnalysisTool
    from workbench.tools.eis.tool import EisTool
    from workbench.tools.drt.tool import DrtTool
    from workbench.tools.ecsa.tool import EcsaTool
    from workbench.tools.it_integrate.tool import ItIntegrateTool
    from workbench.tools.txt_xlsx.tool import TxtXlsxTool
    from workbench.tools.ink_calc.tool import InkCalcTool
    from workbench.tools.ftir.tool import FtirTool

    return [
        LsvAnalysisTool(),
        EisTool(),
        DrtTool(),
        EcsaTool(),
        ItIntegrateTool(),
        TxtXlsxTool(),
        InkCalcTool(),
        FtirTool(),
    ]
