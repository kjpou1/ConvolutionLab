from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory="templates")

FASTAPI_REST_URL = "http://127.0.0.1:8008/predict"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main FastHTML form.
    """
    return templates.TemplateResponse("fasthtml_index.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...),
):
    """
    Handle form submission and call the FastAPI REST service.
    """
    payload = {
        "payload": {
            "data": {
                "gender": gender,
                "race_ethnicity": ethnicity,
                "parental_level_of_education": parental_level_of_education,
                "lunch": lunch,
                "test_preparation_course": test_preparation_course,
                "reading_score": reading_score,
                "writing_score": writing_score,
            }
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(FASTAPI_REST_URL, json=payload)
            response_data = response.json()

        if response.status_code == 200:
            result = response_data.get("data", {}).get("math_score", "N/A")
            return templates.TemplateResponse(
                "fasthtml_result.html", {"request": request, "result": result}
            )
        else:
            error_message = response_data.get("message", "Unknown error occurred.")
            return templates.TemplateResponse(
                "fasthtml_result.html", {"request": request, "error": error_message}
            )
    except Exception as e:
        return templates.TemplateResponse(
            "fasthtml_result.html",
            {
                "request": request,
                "error": f"Failed to connect to prediction service: {str(e)}",
            },
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
