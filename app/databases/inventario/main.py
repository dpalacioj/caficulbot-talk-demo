from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database import get_db, Inventario

class InventarioCreate(BaseModel):
    producto: str
    cantidad: int
app = FastAPI()

@app.post("/inventarioingresar/")
def ingresar_inventario(inventario: InventarioCreate, db: Session = Depends(get_db)):
    """Ingresar un inventario"""
    db_inventario = Inventario(**inventario.dict())
    db.add(db_inventario)
    db.commit()
    return {"ok": "inventario ingresado"}

@app.get("/inventarioconsultar/")
def obtener_inventario(
    producto: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Obtener inventario"""
    query = db.query(Inventario.cantidad)
    
    if producto:
        query = query.filter(Inventario.producto == producto)
    
    cantidadtotal = [cantidad[0] for cantidad in query.all()]
    return sum(cantidadtotal)