from copy import deepcopy
from multiprocessing import Manager
import time
import numpy as np
from bacteria import bacteria
from fastaReader import fastaReader

def calcular_tumbo_adaptativo(iteracion_actual, iteraciones_totales, tumbo_inicial, fitness_max):
    """Tumbo adaptativo"""
    enfriamiento = 0.5 ** (iteracion_actual / iteraciones_totales)
    reduccion_outlier = 1.0
    if fitness_max > 500:  # Umbral ajustado
        reduccion_outlier = max(0.01, 500 / fitness_max)  # Limita reducción mínima al 1%
    
    tumbo_actual = tumbo_inicial * enfriamiento * reduccion_outlier
    return max(min(tumbo_actual, tumbo_inicial), 5)  # Asegura 5 ≤ tumbo ≤ tumbo_inicial

if __name__ == "__main__":
    # Parámetros iniciales
    numeroDeBacterias = 6
    iteraciones = 5
    tumbo_inicial = 300
    nado = 3
    
    # Parámetros de atracción/repulsión ajustados
    dAttr = 0.15
    wAttr = 0.005 
    hRep = dAttr
    wRep = 0.003  
    
    # Lectura de secuencias
    secuencias = fastaReader().seqs
    names = fastaReader().names
    
    #hace todas las secuencias listas de caracteres
    for i in range(len(secuencias)):
        #elimina saltos de linea
        secuencias[i] = list(secuencias[i])
        
        
    numSec = len(secuencias)
    print(f"Número de secuencias: {numSec}")

    manager = Manager()
    poblacion = manager.list([deepcopy(secuencias) for _ in range(numeroDeBacterias)])
    
    operadorBacterial = bacteria(numeroDeBacterias)
    veryBest = [None, -np.inf, None]
    globalNFE = 0
    start_time = time.time()

    for it in range(iteraciones):
        print(f"\n--- Iteración {it+1}/{iteraciones} ---")
        
        # 1. Cálculo de tumbo adaptativo
        current_max_fitness = veryBest[1] if veryBest[0] is not None else 0
        tumbo_actual = calcular_tumbo_adaptativo(it, iteraciones, tumbo_inicial, current_max_fitness)
        tumbo_int = int(tumbo_actual)
        print(f"Tumbo actual: {tumbo_int:.2f}")
        
        # 2. Movimiento bacteriano controlado
        operadorBacterial.tumbo(numSec, poblacion, tumbo_int)
        operadorBacterial.cuadra(numSec, poblacion)
        
        # 3. Evaluación con controles
        try:
            operadorBacterial.creaGranListaPares(poblacion)
            operadorBacterial.evaluaBlosum()
            
            # Versión segura de creación de tablas
            operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, hRep, wRep)
            operadorBacterial.creaTablaInteraction()
            
            # Función de fitness balanceada
            operadorBacterial.creaTablaFitness()
            
            # Validación de resultados
            if not hasattr(operadorBacterial, 'tablaFitness') or any(f is None for f in operadorBacterial.tablaFitness):
                raise ValueError("Fitness no calculado correctamente")
                
        except Exception as e:
            print(f"Error en evaluación: {str(e)}")
            print("Reiniciando población...")
            poblacion = manager.list([deepcopy(secuencias) for _ in range(numeroDeBacterias)])
            continue
        
        # 4. Selección con verificación
        
        globalNFE += operadorBacterial.getNFE()
        bestIdx, bestFitness = operadorBacterial.obtieneBest(globalNFE)
        
        if bestIdx is not None and bestFitness is not None:
            print(f"Mejor fitness actual: {bestFitness:.2f}")
            if bestFitness > veryBest[1]:
                veryBest = [bestIdx, bestFitness, deepcopy(poblacion[bestIdx])]
                print(f"Nuevo mejor global: {bestFitness:.2f}")
        
        # 5. Reemplazo seguro
        if veryBest[0] is not None:
            operadorBacterial.replaceWorst(poblacion, veryBest[0])
        
        operadorBacterial.resetListas(numeroDeBacterias)

    # Resultados finales
    print("\n--- Resultados Finales ---")
    if veryBest[0] is not None:
        print(f"Mejor fitness: {veryBest[1]:.2f}")
        print(f"Mejor secuencia: {veryBest[2]}")
    else:
        print("No se encontraron soluciones válidas")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")