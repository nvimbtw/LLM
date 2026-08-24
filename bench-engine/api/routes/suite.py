from fastapi import APIRouter
router = APIRouter(tags=["suite"])
@router.get("/suite")
def suite():
    return {"suites": ["language-v3"], "default": "language-v3", "language_v3": {"version": 3, "prompts": 32, "grammar_pairs": 30}}
