from copy import deepcopy
from multiprocessing import Manager
import time
import numpy as np
from bacteria import bacteria
from fastaReader import fastaReader

def calcular_tumbo_adaptativo(iteracion_actual, iteraciones_totales, tumbo_inicial, fitness_max):
    """Tumbo adaptativo"""
    enfriamiento = 0.5 ** (iteracion_actual / iteraciones_totales)
    reduccion_anomalia = 1.0
    if fitness_max > 500:  # Umbral ajustado
        reduccion_anomalia = max(0.01, 500 / fitness_max)  # Limita reducción mínima al 1%
    
    tumbo_actual = tumbo_inicial * enfriamiento * reduccion_anomalia
    return max(min(tumbo_actual, tumbo_inicial), 5)  # Asegura 5 ≤ tumbo ≤ tumbo_inicial

def corregir_anomalia(poblacion, idx, secuencias, veryBest, operadorBacterial):
    """Corrige una bacteria con fitness anómalo"""
    try:
        # 1. Preparar secuencia de respaldo
        if veryBest[0] is not None and veryBest[1] < 100:
            secuencia_respaldo = deepcopy(veryBest[2])
        else:
            secuencia_respaldo = deepcopy(secuencias[idx % len(secuencias)])
        
        if isinstance(secuencia_respaldo, str):
            secuencia_respaldo = list(secuencia_respaldo)
        
        # 2. Aplicar corrección
        poblacion[idx] = secuencia_respaldo
        
        # 3. Recalcular fitness
        operadorBacterial.creaGranListaPares([poblacion[idx]])
        operadorBacterial.evaluaBlosum()
        nuevo_fitness = operadorBacterial.tablaFitness[0]
        
        if nuevo_fitness > 100:
            raise ValueError("Fitness corregido sigue siendo anómalo")
            
        return (True, nuevo_fitness)
        
    except Exception as e:
        print(f"Error en corrección: {str(e)}")
        poblacion[idx] = ['A','T','G','C']
        return (False, 0)

if __name__ == "__main__":
    # Parámetros iniciales
    numeroDeBacterias = 6
    iteraciones = 5
    tumbo_inicial = 300
    nado = 3
    dAttr = 0.15
    wAttr = 0.005 
    hRep = dAttr
    wRep = 0.003  
    UMBRAL_FITNESS = 100

    # Lectura y preparación de secuencias
    secuencias = [list(seq) if isinstance(seq, str) else seq for seq in fastaReader().seqs]
    names = fastaReader().names
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
        print(f"Tumbo actual: {int(tumbo_actual)}")
        
        # 2. Movimiento bacteriano
        operadorBacterial.tumbo(numSec, poblacion, int(tumbo_actual))
        operadorBacterial.cuadra(numSec, poblacion)
        
        # 3. Evaluación con controles
        try:
            operadorBacterial.creaGranListaPares(poblacion)
            operadorBacterial.evaluaBlosum()
            operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, hRep, wRep)
            operadorBacterial.creaTablaInteraction()
            operadorBacterial.creaTablaFitness()
            
            if not hasattr(operadorBacterial, 'tablaFitness') or any(f is None for f in operadorBacterial.tablaFitness):
                raise ValueError("Fitness no calculado correctamente")
                
        except Exception as e:
            print(f"Error en evaluación: {str(e)}")
            poblacion = manager.list([deepcopy(secuencias) for _ in range(numeroDeBacterias)])
            continue
        
        # 4. Manejo de anomalías
        globalNFE += operadorBacterial.getNFE()
        bestIdx, bestFitness = operadorBacterial.obtieneBest(globalNFE)
        
        if bestFitness > UMBRAL_FITNESS:
            print(f"\n¡ANOMALÍA DETECTADA! Fitness: {bestFitness}")
            exito, nuevo_fitness = corregir_anomalia(poblacion, bestIdx, secuencias, veryBest, operadorBacterial)
            
            if exito:
                print(f"Corrección exitosa. Nuevo fitness: {nuevo_fitness}")
                bestFitness = nuevo_fitness
                # Recalcular todo con la bacteria corregida
                operadorBacterial.creaGranListaPares(poblacion)
                operadorBacterial.evaluaBlosum()
                operadorBacterial.creaTablaFitness()
            else:
                print("¡Corrección falló! Descartando bacteria.")
                bestFitness = 0
        
        # 5. Actualización de veryBest
        if bestIdx is not None and bestFitness is not None:
            print(f"Mejor fitness actual: {bestFitness:.2f}")
            if veryBest[0] is None or bestFitness > veryBest[1]:
                veryBest = [bestIdx, bestFitness, deepcopy(poblacion[bestIdx])]
                print(f"Nuevo mejor global: {bestFitness:.2f}")
        
        # 6. Reemplazo seguro
        if veryBest[0] is not None:
            operadorBacterial.replaceWorst(poblacion, veryBest[0])
        
        operadorBacterial.resetListas(numeroDeBacterias)

    # Resultados finales
    print("\n--- Resultados Finales ---")
    if veryBest[0] is not None:
        print(f"Mejor fitness encontrado: {veryBest[1]:.2f}")
        print(f"Índice de la mejor bacteria: {veryBest[0]}")
        print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
        print(f"Evaluaciones de función (NFE): {globalNFE}")
    else:
        print("No se encontraron soluciones válidas. Posibles causas:")
        print("- Todas las corridas produjeron valores anómalos")
        print("- Error en el cálculo inicial del fitness")
        print("- Parámetros demasiado restrictivos")
