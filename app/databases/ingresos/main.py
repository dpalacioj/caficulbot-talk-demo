from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database import get_db, Ingreso

class IngresoCreate(BaseModel):
    año: int
    mes: int
    monto: float

app = FastAPI()

@app.post("/ingresosingresar/")
def ingresar_ingreso(ingreso: IngresoCreate, db: Session = Depends(get_db)):
    """Ingresar un ingreso"""
    db_ingreso = Ingreso(**ingreso.dict())
    db.add(db_ingreso)
    db.commit()
    return {"ok": "ingreso ingresado"}

@app.get("/ingresosconsultar/")
def obtener_ingresos(
    año: Optional[int] = None,
    mes: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Obtener ingresos"""
    query = db.query(Ingreso.monto)
    
    if año:
        query = query.filter(Ingreso.año == año)
    if mes:
        query = query.filter(Ingreso.mes == mes)
    
    montos = [monto[0] for monto in query.all()]
    return sum(montos)