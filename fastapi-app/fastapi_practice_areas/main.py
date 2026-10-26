from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Consultant, PracticeArea
from schemas import (
    ConsultantCreate,
    ConsultantOut,
    PracticeAreaCreate,
    PracticeAreaOut,
    ConsultantUpdate,
    PracticeAreaWithConsultants,
)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Practice Area and Consultant API")


@app.post("/practice-areas", response_model=PracticeAreaOut, status_code=201)
def create_practice_area(
    practice_area: PracticeAreaCreate,
    db: Session = Depends(get_db),
):
    item = PracticeArea(**practice_area.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.post(
    "/practice-areas/{practice_area_id}/consultants",
    response_model=ConsultantOut,
    status_code=201,
)
def add_consultant(
    practice_area_id: int,
    consultant: ConsultantCreate,
    db: Session = Depends(get_db),
):
    parent = db.get(PracticeArea, practice_area_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="Practice area not found")

    item = Consultant(
        **consultant.model_dump(),
        practice_area_id=practice_area_id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get(
    "/practice-areas/{practice_area_id}",
    response_model=PracticeAreaWithConsultants,
)
def get_practice_area(
    practice_area_id: int,
    designation: str | None = Query(default=None),
    sort: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    parent = db.get(PracticeArea, practice_area_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="Practice area not found")

    consultants = list(parent.consultants)

    if designation is not None and designation.strip():
        consultants = [c for c in consultants if c.designation == designation]

    if sort is not None:
        if sort not in ("name", "consultation_fee"):
            raise HTTPException(status_code=404, detail="Invalid sort field")
        consultants.sort(key=lambda c: getattr(c, sort))

    return {
        "id": parent.id,
        "name": parent.name,
        "floor_number": parent.floor_number,
        "consultants": consultants,
    }


@app.put(
    "/practice-areas/{practice_area_id}/consultants/{consultant_id}",
    response_model=ConsultantOut
)
def update_consultant(
    practice_area_id: int,
    consultant_id: int,
    data: ConsultantUpdate,
    db: Session = Depends(get_db),
):
    consultant = db.query(Consultant).filter(
        Consultant.id == consultant_id,
        Consultant.practice_area_id == practice_area_id
    ).first()

    if consultant is None:
        raise HTTPException(
            status_code=404,
            detail="Consultant not found in this practice area"
        )

    update_data = data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(consultant, field, value)

    db.commit()
    db.refresh(consultant)

    return consultant


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
