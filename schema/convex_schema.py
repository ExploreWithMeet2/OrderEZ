from typing import Annotated
from pydantic import BaseModel, Field


class ConvexRequest(BaseModel):
    module: Annotated[str, Field(..., description="Module Name")]
    func: Annotated[str, Field(..., description="Function Inside Module")]
    isQuery: Annotated[bool, Field(..., description="True = query, False = mutation")]
    data: Annotated[dict, Field(None, description="Data to Post")]
    args: Annotated[dict, Field(None, description="Arguments to Post")]
    returnDf: Annotated[bool, Field(True, description="True = Df, False = Json")]


class ConvexResponse(BaseModel):
    success: bool
    data: dict | list = None
    error: str | None = None
