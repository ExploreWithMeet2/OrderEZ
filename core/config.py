from os import getenv
from dotenv import load_dotenv

load_dotenv()


class Config:
    convex_url: str = getenv("CONVEX_URL")
    b_a = ["jh70ns6tjkyrj3n0va5s292c017vdq98","jh78y4mnxyxsmhj2tbwcwqxa497vdb5p","jh77v9c3j5npptckcm5cgrc6zx7vc376","jh7dkhq6xrccfe19rmwb5h0y657vdq67","jh7951gzjejh0mxr1frcxkwz157vdccd","jh7e68q8t6ghhr6mac0x29h0617vdbc0"]