"""Tool registry. Adding a tool = add its class here."""
from __future__ import annotations

from workbench.core.tool_base import Tool


def build_tools() -> list[Tool]:
    # Imported lazily so a failure in one tool doesn't block the whole app.
    from workbench.tools.lsv.tool import LsvTool
    from workbench.tools.eis.tool import EisTool
    from workbench.tools.ecsa.tool import EcsaTool
    from workbench.tools.it_integrate.tool import ItIntegrateTool
    from workbench.tools.txt_xlsx.tool import TxtXlsxTool
    from workbench.tools.ink_calc.tool import InkCalcTool

    return [
        LsvTool(),
        EisTool(),
        EcsaTool(),
        ItIntegrateTool(),
        TxtXlsxTool(),
        InkCalcTool(),
    ]
