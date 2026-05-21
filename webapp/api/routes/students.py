"""GET /students — student list for dropdown.
GET /students/{index} — single student info.
"""

from fastapi import APIRouter, HTTPException

from api.data_manager import data_manager

router = APIRouter()


@router.get("/students")
def list_students():
    """Return all test students for the UI dropdown."""
    if not data_manager.ready:
        raise HTTPException(503, "Data not loaded yet")
    students = data_manager.get_student_list()
    return {
        "count": len(students),
        "students": students,
    }


@router.get("/students/{index}")
def get_student(index: int):
    """Return display info for a single student."""
    if not data_manager.ready:
        raise HTTPException(503, "Data not loaded yet")
    if index < 0 or index >= data_manager.n_students:
        raise HTTPException(404, f"Student index {index} out of range (0–{data_manager.n_students - 1})")
    return data_manager.get_student_info(index)
