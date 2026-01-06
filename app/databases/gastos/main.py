from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database import get_db, Gasto

# Modelo para crear gasto
class GastoCreate(BaseModel):
    año: int
    mes: int
    categoria: Optional[str] = None
    monto: float

# App FastAPI
app = FastAPI()

@app.post("/gastosingresar/")
def ingresar_gasto(gasto: GastoCreate, db: Session = Depends(get_db)):
    """Ingresar un gasto"""
    db_gasto = Gasto(**gasto.dict())
    db.add(db_gasto)
    db.commit()
    return {"ok": "gasto ingresado"}

@app.get("/gastosconsultar/")
def obtener_gastos(
    año: Optional[int] = None,
    mes: Optional[int] = None,
    categoria: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Obtener gastos"""
    query = db.query(Gasto.monto)
    
    if año:
        query = query.filter(Gasto.año == año)
    if mes:
        query = query.filter(Gasto.mes == mes)
    if categoria:
        query = query.filter(Gasto.categoria.ilike(f"%{categoria}%"))
    
    montos = [monto[0] for monto in query.all()]
    return sum(montos)