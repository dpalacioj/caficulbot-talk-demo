from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database import get_db, Cosecha

class CosechaCreate(BaseModel):
    año: int
    mes: int
    cantidad: int
app = FastAPI()

@app.post("/cosechaingresar/")
def ingresar_cosecha(cosecha: CosechaCreate, db: Session = Depends(get_db)):
    """Ingresar un cosecha"""
    db_cosecha = Cosecha(**cosecha.dict())
    db.add(db_cosecha)
    db.commit()
    return {"ok": "cosecha ingresado"}

@app.get("/cosechaconsultar/")
def obtener_cosecha(
    año: Optional[int] = None,
    mes: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Obtener cosecha"""
    query = db.query(Cosecha.monto)
    
    if año:
        query = query.filter(Cosecha.año == año)
    if mes:
        query = query.filter(Cosecha.mes == mes)
    
    montos = [monto[0] for monto in query.all()]
    return sum(montos)