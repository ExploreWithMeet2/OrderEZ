import httpx
import pandas as pd
from utils.returnFormat import returnFormat
from core.config import Config
from schema.convex_schema import ConvexRequest

CONVEX_URL = Config.convex_url


async def call_convex(c_req: ConvexRequest) -> dict:
    if not CONVEX_URL:
        return returnFormat("error", "Convex API URL not configured in environment")

    endpoint = "query" if c_req.isQuery else "mutation"
    url = f"{CONVEX_URL}/api/{endpoint}"
    payload = {"path": f"{c_req.module}:{c_req.func}", "args": c_req.args or {}}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            print("Convex Result:", result)

            if result.get("status") == "error":
                return returnFormat(
                    "error", result.get("errorMessage", "Convex Request: Unknown error")
                )

            data = result.get("value")
            if c_req.returnDf:
                return returnFormat(
                    "success", "Convex Request: Successfull", pd.DataFrame(data)
                )
            else:
                return returnFormat("success", "Convex Request: Successfull", data)

    except httpx.RequestError as e:
        return returnFormat("error", f"Convex API request failed: {str(e)}")
    except Exception as e:
        return returnFormat("error", f"Unexpected error: {str(e)}")
