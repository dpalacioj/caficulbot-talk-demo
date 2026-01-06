#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Iniciando CaficulBot - Entorno Local${NC}"
echo -e "${GREEN}========================================${NC}"

cleanup() {
    echo -e "\n${YELLOW}Deteniendo todos los servicios...${NC}"
    
    pkill -P $$
    
    for port in 8000 8001 8002 8003 8004 8501; do
        lsof -ti:$port | xargs kill -9 2>/dev/null
    done
    
    echo -e "${GREEN}Todos los servicios detenidos.${NC}"
    exit 0
}

trap cleanup EXIT INT TERM

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creando entorno virtual...${NC}"
    python3 -m venv venv
fi

echo -e "${YELLOW}Activando entorno virtual...${NC}"
source venv/bin/activate

echo -e "${YELLOW}Verificando dependencias...${NC}"
pip install -q --upgrade pip

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU NVIDIA detectada. Instalando PyTorch con soporte CUDA...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    echo -e "${YELLOW}No se detectó GPU NVIDIA. Instalando PyTorch para CPU...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

pip install -q -r requirements.txt

LOG_DIR="logs"
mkdir -p $LOG_DIR

rm -f $LOG_DIR/*.log

echo -e "${GREEN}Iniciando servicios de base de datos...${NC}"

echo -e "${YELLOW}  → Iniciando servicio de Inventario en puerto 8001...${NC}"
cd databases/inventario
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload > ../../$LOG_DIR/inventario.log 2>&1 &
INVENTORY_PID=$!
cd ../..
sleep 2

echo -e "${YELLOW}  → Iniciando servicio de Gastos en puerto 8002...${NC}"
cd databases/gastos
python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload > ../../$LOG_DIR/gastos.log 2>&1 &
EXPENSES_PID=$!
cd ../..
sleep 2

echo -e "${YELLOW}  → Iniciando servicio de Cosecha en puerto 8003...${NC}"
cd databases/cosecha
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload > ../../$LOG_DIR/cosecha.log 2>&1 &
PRODUCTION_PID=$!
cd ../..
sleep 2

echo -e "${YELLOW}  → Iniciando servicio de Ingresos en puerto 8004...${NC}"
cd databases/ingresos
python -m uvicorn main:app --host 0.0.0.0 --port 8004 --reload > ../../$LOG_DIR/ingresos.log 2>&1 &
INCOME_PID=$!
cd ../..
sleep 2

echo -e "${YELLOW}  → Iniciando API principal en puerto 8000...${NC}"
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload > $LOG_DIR/api.log 2>&1 &
API_PID=$!
sleep 15

echo -e "${YELLOW}  → Iniciando interfaz web Streamlit en puerto 8501...${NC}"
streamlit run web.py --server.port 8501 --server.address 0.0.0.0 > $LOG_DIR/streamlit.log 2>&1 &
WEB_PID=$!
sleep 15

echo -e "\n${GREEN}Verificando estado de los servicios...${NC}"

check_service() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "  ✓ ${GREEN}$name está activo en puerto $port${NC}"
        return 0
    else
        echo -e "  ✗ ${RED}$name NO está activo en puerto $port${NC}"
        return 1
    fi
}

sleep 2

ALL_GOOD=true
check_service 8001 "Servicio Inventario" || ALL_GOOD=false
check_service 8002 "Servicio Gastos" || ALL_GOOD=false
check_service 8003 "Servicio Cosecha" || ALL_GOOD=false
check_service 8004 "Servicio Ingresos" || ALL_GOOD=false
check_service 8000 "API Principal" || ALL_GOOD=false
check_service 8501 "Interfaz Web" || ALL_GOOD=false

echo -e "\n${GREEN}========================================${NC}"
if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✓ Todos los servicios están activos!${NC}"
    echo -e "\n${GREEN}URLs disponibles:${NC}"
    echo -e "  • API Principal:        ${YELLOW}http://localhost:8000${NC}"
    echo -e "  • Interfaz Web:         ${YELLOW}http://localhost:8501${NC}"
    echo -e "  • API Inventario:       ${YELLOW}http://localhost:8001${NC}"
    echo -e "  • API Gastos:           ${YELLOW}http://localhost:8002${NC}"
    echo -e "  • API Cosecha:          ${YELLOW}http://localhost:8003${NC}"
    echo -e "  • API Ingresos:         ${YELLOW}http://localhost:8004${NC}"
    echo -e "\n${GREEN}Documentación APIs (Swagger):${NC}"
    echo -e "  • API Principal:        ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "  • API Inventario:       ${YELLOW}http://localhost:8001/docs${NC}"
    echo -e "  • API Gastos:           ${YELLOW}http://localhost:8002/docs${NC}"
    echo -e "  • API Cosecha:          ${YELLOW}http://localhost:8003/docs${NC}"
    echo -e "  • API Ingresos:         ${YELLOW}http://localhost:8004/docs${NC}"
else
    echo -e "${RED}⚠ Algunos servicios no se iniciaron correctamente${NC}"
    echo -e "${YELLOW}Revisa los logs en el directorio '$LOG_DIR' para más detalles${NC}"
fi
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Logs disponibles en:${NC}"
echo -e "  • tail -f $LOG_DIR/api.log         (API principal)"
echo -e "  • tail -f $LOG_DIR/streamlit.log   (Interfaz web)"
echo -e "  • tail -f $LOG_DIR/inventario.log  (Servicio inventario)"
echo -e "  • tail -f $LOG_DIR/gastos.log      (Servicio gastos)"
echo -e "  • tail -f $LOG_DIR/cosecha.log     (Servicio cosecha)"
echo -e "  • tail -f $LOG_DIR/ingresos.log    (Servicio ingresos)"

echo -e "\n${YELLOW}Presiona Ctrl+C para detener todos los servicios${NC}\n"

while true; do
    sleep 1
done