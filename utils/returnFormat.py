from typing import Literal


def returnFormat(type_: Literal["success", "error"], message: str, data=None) -> dict:
    return {"type": type_, "data": data, "message": message}
