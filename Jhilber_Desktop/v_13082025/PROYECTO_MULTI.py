import pandas as pd
import pulp
import sys
import time
sys.stdout = open("salida_MULTI.txt", "w", encoding='utf-8')
# ========================================
# SECCIÓN 1: DEFINICIÓN DE FUNCIONES
# ========================================
def cargar_datos():
    DATOS_LINEA = pd.read_csv('DATA_LINEAS.csv')  
    DATOS_NODOS = pd.read_csv('DATA_NODOS.csv')   
    # CONJUNTOS BÁSICOS
    PSI_N = list(set(DATOS_LINEA['u'].unique()) | set(DATOS_LINEA['v'].unique()))
    PSI_LN = list(set(DATOS_NODOS['Node'].unique()))
    PSI_SS = list((set(DATOS_LINEA['u'].unique()) | set(DATOS_LINEA['v'].unique())) - (set(DATOS_NODOS['Node'].unique())))
    UPSILON = [(int(row['u']), int(row['v'])) for _, row in DATOS_LINEA.iterrows()]
    return DATOS_LINEA, DATOS_NODOS, PSI_N, PSI_LN, PSI_SS, UPSILON

def identificar_Transformador_Principal(PSI_SS, PSI_LN, UPSILON):
    """Identifica las redes de transformadores"""
    TransformadoresPrincipal_Transformadores = {}
    # Crear grafo de adyacencia
    grafo = {}
    for u, v in UPSILON:
        if u not in grafo:
            grafo[u] = []
        if v not in grafo:
            grafo[v] = []
        grafo[u].append(v)
        grafo[v].append(u)
    for ss in PSI_SS:
        TransformadoresPrincipal_Transformadores[ss] = []
        cola = [ss]
        visitados_local = {ss}
        while cola:
            nodo_actual = cola.pop(0)
            if nodo_actual in grafo:
                for vecino in grafo[nodo_actual]:
                    if vecino not in visitados_local:
                        visitados_local.add(vecino)
                        if vecino in PSI_LN:
                            TransformadoresPrincipal_Transformadores[ss].append(vecino)
                        cola.append(vecino)
    return TransformadoresPrincipal_Transformadores

def crear_diccionarios_datos(DATOS_LINEA, DATOS_NODOS, PSI_N, UPSILON):
    """Crea los diccionarios con los datos necesarios"""
    # CONJUNTOS ADYACENTES
    PSI_ADY = {}
    for nodo in PSI_N:
        PSI_ADY[nodo] = []
    for u, v in UPSILON:
        PSI_ADY[u].append(v)
        PSI_ADY[v].append(u)
    # DATOS PARA CADA LINEA
    LAMBDA_DICT = {}
    TAU_RP_DICT = {}
    TAU_SW_DICT = {}
    for _, row in DATOS_LINEA.iterrows():
        u, v = int(row['u']), int(row['v'])
        for (i, j) in [(u, v), (v, u)]:
            LAMBDA_DICT[(i, j)] = row['lambda']
            TAU_RP_DICT[(i, j)] = row['Tao_RP']
            TAU_SW_DICT[(i, j)] = row['Tao_SW']
    # DATOS PARA CADA NODO
    N_C_DICT = {}
    L_MW_DICT = {}
    for _, row in DATOS_NODOS.iterrows():
        nodo = int(row['Node'])
        N_C_DICT[nodo] = row['NC']
        L_MW_DICT[nodo] = row['L_MW']
    return PSI_ADY, LAMBDA_DICT, TAU_RP_DICT, TAU_SW_DICT, N_C_DICT, L_MW_DICT

def resolver_flujos_por_red(PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES, UPSILON, PSI_ADY):
    """Resuelve los problemas de flujo para cada red"""
    resultados_por_red = {}
    f_tilde = {}
    f_hat = {}
    for ss in PSI_SS:
        # Nodos de la red actual
        nodos_red = [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]
        # Líneas de la red actual
        lineas_red = []
        for u, v in UPSILON:
            if u in nodos_red and v in nodos_red:
                lineas_red.append((u, v))
        # Inicializar estructuras para esta red
        f_tilde[ss] = {}
        f_hat[ss] = {}
        resultados_por_red[ss] = []
        # Resolver para cada transformador de distribución en esta red
        for s in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
            # Crear problema de optimización
            prob = pulp.LpProblem(f"Flujo_Red_{ss}_Nodo_{s}", pulp.LpMinimize)
            # Variables de decisión solo para nodos de la red
            f_vars = {}
            g_vars = {}
            
            # Variables f_ijs solo para líneas de la red
            for i, j in lineas_red:
                f_vars[(i, j, s)] = pulp.LpVariable(f"f_{i}{j}{s}", 0, 1)
                f_vars[(j, i, s)] = pulp.LpVariable(f"f_{j}{i}{s}", 0, 1)
            
            # Variable de generación solo para el transformador principal de la red
            g_vars[(ss, s)] = pulp.LpVariable(f"g_{ss}{s}", 0, 1)
            
            # Función objetivo
            objetivo = [var for var in f_vars.values()]
            prob += pulp.lpSum(objetivo)
            
            # Restricciones para cada nodo de la red
            for i in nodos_red:
                flujo_neto = []
                
                # Solo considerar vecinos dentro de la red
                for j in PSI_ADY[i]:
                    if j in nodos_red:
                        if (i, j, s) in f_vars and (j, i, s) in f_vars:
                            flujo_neto.append(f_vars[(i, j, s)])
                            flujo_neto.append(-f_vars[(j, i, s)])
                
                # Definir L_is y g_is
                L_is = 1 if i == s else 0
                g_is = g_vars[(ss, s)] if i == ss else 0
                
                # Restricción de balance
                if flujo_neto:
                    prob += pulp.lpSum(flujo_neto) == g_is - L_is
                else:
                    prob += 0 == g_is - L_is
            
            # Resolver
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Almacenar soluciones
            if prob.status == pulp.LpStatusOptimal:
                if s not in f_tilde:
                    f_tilde[s] = {}
                    f_hat[s] = {}
                
                for key, var in f_vars.items():
                    valor = pulp.value(var)
                    i, j, nodo_s = key
                    
                    f_tilde[s][(i, j)] = valor if valor > 1e-6 else 0
                    f_hat[s][(i, j)] = valor if valor > 1e-6 else 0
                    
                    if valor > 1e-6:
                        resultados_por_red[ss].append({
                            'Red': ss,
                            'Nodo_s': nodo_s,
                            'Variable': f'f_{i}{j}{nodo_s}',
                            'Valor': round(valor, 6)
                        })
    
    return f_tilde, f_hat, resultados_por_red

def calcular_indices_confiabilidad_multi(PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES, PSI_ADY, f_tilde, f_hat, UPSILON, LAMBDA_DICT, TAU_RP_DICT, TAU_SW_DICT):
    """Calcula índices de confiabilidad para sistema multi-red"""
    indices_por_red = {}
    for ss in PSI_SS:
        # Calcular N_ij^CB para esta red
        N_CB = {}
        for i in [ss]:  # Transformador principal de la red
            for j in PSI_ADY[i]:
                if j in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:  # Solo vecinos de la misma red
                    N_CB[(i, j)] = 0
                    for s_prime in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
                        suma_interna = 0
                        for r in PSI_ADY[s_prime]:
                            if r in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:  # Solo dentro de la red
                                f_rss = f_tilde[s_prime].get((r, s_prime), 0)
                                lambda_rs = LAMBDA_DICT.get((r, s_prime), 0)
                                suma_interna += f_rss * lambda_rs
                        
                        f_ijs = f_tilde[s_prime].get((i, j), 0)
                        N_CB[(i, j)] += suma_interna * f_ijs
        
        # Calcular D_ij^CB,SW para esta red
        D_CB_SW = {}
        for i in [ss]:
            for j in PSI_ADY[i]:
                if j in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
                    D_CB_SW[(i, j)] = 0
                    for s_prime in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
                        suma_interna = 0
                        for r in PSI_ADY[s_prime]:
                            if r in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
                                f_rss = f_hat[s_prime].get((r, s_prime), 0)
                                lambda_rs = LAMBDA_DICT.get((r, s_prime), 0)
                                tau_SW_rs = TAU_SW_DICT.get((r, s_prime), 0)
                                suma_interna += f_rss * lambda_rs * tau_SW_rs
                        
                        f_ijs = f_hat[s_prime].get((i, j), 0)
                        D_CB_SW[(i, j)] += suma_interna * f_ijs
        
        # Calcular índices para cada transformador de distribución de la red
        indices_red = []
        
        for s in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
            # N_s^RP
            N_s_RP = 0
            for i, j in UPSILON:
                if i in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss] and j in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
                    f_ijs_tilde = f_tilde[s].get((i, j), 0)
                    f_jis_tilde = f_tilde[s].get((j, i), 0)
                    lambda_ij = LAMBDA_DICT[(i, j)]
                    N_s_RP += (f_ijs_tilde + f_jis_tilde) * lambda_ij
            
            # N_s^T
            N_s_T = 0
            for i in [ss]:
                for j in PSI_ADY[i]:
                    if j in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss] and (i, j) in N_CB:
                        f_ijs_tilde = f_tilde[s].get((i, j), 0)
                        f_jis_tilde = f_tilde[s].get((j, i), 0)
                        N_s_T += N_CB[(i, j)] * (f_ijs_tilde + f_jis_tilde)
            
            # N_s^SW
            N_s_SW = N_s_T - N_s_RP
            
            # D_s^RP
            D_s_RP = 0
            for i, j in UPSILON:
                if i in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss] and j in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
                    f_ijs_tilde = f_tilde[s].get((i, j), 0)
                    f_jis_tilde = f_tilde[s].get((j, i), 0)
                    lambda_ij = LAMBDA_DICT[(i, j)]
                    tau_RP_ij = TAU_RP_DICT[(i, j)]
                    D_s_RP += (f_ijs_tilde + f_jis_tilde) * lambda_ij * tau_RP_ij
            
            # D_s^T,SW
            D_s_T_SW = 0
            for i in [ss]:
                for j in PSI_ADY[i]:
                    if j in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss] and (i, j) in D_CB_SW:
                        f_ijs_hat = f_hat[s].get((i, j), 0)
                        f_jis_hat = f_hat[s].get((j, i), 0)
                        D_s_T_SW += D_CB_SW[(i, j)] * (f_ijs_hat + f_jis_hat)
            
            # D̄_s^SW
            D_bar_s_SW = 0
            for i, j in UPSILON:
                if i in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss] and j in [ss] + TRANSFORMADORESPRINCIPALES_TRANSFORMADORES[ss]:
                    f_ijs_hat = f_hat[s].get((i, j), 0)
                    f_jis_hat = f_hat[s].get((j, i), 0)
                    lambda_ij = LAMBDA_DICT[(i, j)]
                    tau_SW_ij = TAU_SW_DICT[(i, j)]
                    D_bar_s_SW += (f_ijs_hat + f_jis_hat) * lambda_ij * tau_SW_ij
            
            # D_s^SW
            D_s_SW = D_s_T_SW - D_bar_s_SW
            
            # CIF_s y CID_s
            CIF_s = N_s_RP + N_s_SW
            CID_s = D_s_RP + D_s_SW
            
            indices_red.append({
                'Red': ss,
                'Nodo': s,
                'N_s_RP': round(N_s_RP, 6),
                'N_s_SW': round(N_s_SW, 6),
                'D_s_RP': round(D_s_RP, 6),
                'D_s_SW': round(D_s_SW, 6),
                'CIF_s': round(CIF_s, 6),
                'CID_s': round(CID_s, 6)
            })
        
        indices_por_red[ss] = indices_red
    
    return indices_por_red

def calcular_indices_sistema_multi(indices_por_red, PSI_SS, PSI_LN, N_C_DICT, L_MW_DICT):
    """Calcula índices del sistema completo (todas las redes)"""
    # Consolidar todos los índices
    todos_indices = []
    for ss in PSI_SS:
        todos_indices.extend(indices_por_red[ss])
    
    # Calcular totales
    total_clientes = sum(N_C_DICT[s] for s in PSI_LN)
    
    # SAIFI del sistema
    numerador_saifi = sum(N_C_DICT[item['Nodo']] * item['CIF_s'] for item in todos_indices)
    SAIFI = numerador_saifi / total_clientes
    
    # SAIDI del sistema
    numerador_saidi = sum(N_C_DICT[item['Nodo']] * item['CID_s'] for item in todos_indices)
    SAIDI = numerador_saidi / total_clientes
    
    # ASAI del sistema
    ASAI = 1 - (SAIDI / 8760)
    ASAI_percent = ASAI * 100
    
    # EENS del sistema
    EENS = sum(item['CID_s'] * L_MW_DICT[item['Nodo']] for item in todos_indices)
    
    return {
        'SAIFI': round(SAIFI, 6),
        'SAIDI': round(SAIDI, 6),
        'ASAI': round(ASAI_percent, 6),
        'EENS': round(EENS, 6),
        'total_clientes': total_clientes,
        'total_nodos': len(PSI_LN),
        'total_redes': len(PSI_SS)
    }

def calcular_indices_por_red_separados(indices_multi, PSI_SS, N_C_DICT, L_MW_DICT):
    """Calcula los índices por red separados"""
    indices_redes_separadas = {}
    
    for ss in PSI_SS:
        # Calcular índices específicos de la red
        nodos_red = [item for item in indices_multi[ss]]
        clientes_red = sum(N_C_DICT[item['Nodo']] for item in nodos_red)
        
        saifi_red = sum(N_C_DICT[item['Nodo']] * item['CIF_s'] for item in nodos_red) / clientes_red
        saidi_red = sum(N_C_DICT[item['Nodo']] * item['CID_s'] for item in nodos_red) / clientes_red
        asai_red = (1 - saidi_red / 8760) * 100
        eens_red = sum(item['CID_s'] * L_MW_DICT[item['Nodo']] for item in nodos_red)
        indices_redes_separadas[ss] = {
            'transformadores_dist': len(nodos_red),
            'clientes': clientes_red,
            'SAIFI': round(saifi_red, 6),
            'SAIDI': round(saidi_red, 6),
            'ASAI': round(asai_red, 6),
            'EENS': round(eens_red, 6)
        }
    
    return indices_redes_separadas

# ========================================
# SECCIÓN 2: EJECUCIÓN DEL PROGRAMA
# ========================================

def ejecutar_analisis_completo():
    """Función que ejecuta todo el análisis sin mostrar resultados"""
    # 1. Cargar datos
    DATOS_LINEA, DATOS_NODOS, PSI_N, PSI_LN, PSI_SS, UPSILON = cargar_datos()
    
    # 2. Identificar transformadores principales
    TRANSFORMADORESPRINCIPALES_TRANSFORMADORES = identificar_Transformador_Principal(PSI_SS, PSI_LN, UPSILON)
    
    # 3. Crear diccionarios de datos
    PSI_ADY, LAMBDA_DICT, TAU_RP_DICT, TAU_SW_DICT, N_C_DICT, L_MW_DICT = crear_diccionarios_datos(
        DATOS_LINEA, DATOS_NODOS, PSI_N, UPSILON)
    
    # 4. Resolver flujos por red
    f_tilde, f_hat, resultados_por_red = resolver_flujos_por_red(
        PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES, UPSILON, PSI_ADY)
    
    # 5. Calcular índices de confiabilidad
    indices_multi = calcular_indices_confiabilidad_multi(
        PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES, PSI_ADY, f_tilde, f_hat, 
        UPSILON, LAMBDA_DICT, TAU_RP_DICT, TAU_SW_DICT)
    
    # 6. Calcular índices del sistema
    indices_sistema_multi = calcular_indices_sistema_multi(
        indices_multi, PSI_SS, PSI_LN, N_C_DICT, L_MW_DICT)
    
    # 7. Calcular índices por red separados
    indices_redes_separadas = calcular_indices_por_red_separados(
        indices_multi, PSI_SS, N_C_DICT, L_MW_DICT)
    
    return (PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES, indices_multi, 
            indices_sistema_multi, indices_redes_separadas)

# ========================================
# SECCIÓN 3: FUNCIONES DE IMPRESIÓN
# ========================================

def imprimir_informacion_redes(PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES):
    """Imprime la información de las redes"""
    print("%" * 120)
    print(f"NUMERO DE REDES: {len(PSI_SS)}")
    print(f"{'TR principal':<15}{'Numero de TR Distribucion':<28}{'Transformadores Distribución'}")
    print("-" * 100)
    for ss, nodos in TRANSFORMADORESPRINCIPALES_TRANSFORMADORES.items():
        nodos_str = ", ".join(map(str, sorted(nodos)))
        count_text = f"{len(nodos)}"
        print(f"{ss:<16}{count_text:<29}{nodos_str}")
    print("%" * 120)

def imprimir_resultados_por_red(indices_multi, PSI_SS):
    """Imprime los resultados por red"""
    print("\n" + "="*120)
    print("RESULTADOS POR RED")
    print("="*120)
    
    for ss in PSI_SS:
        print(f"\n--- RED {ss} ---")
        print(f"{'Nodo':<6} {'N_s^RP':<10} {'N_s^SW':<10} {'D_s^RP':<10} {'D_s^SW':<10} {'CIF_s':<10} {'CID_s':<10}")
        print("-" * 70)
        
        for item in indices_multi[ss]:
            print(f"{item['Nodo']:<6} {item['N_s_RP']:<10} {item['N_s_SW']:<10} "
                  f"{item['D_s_RP']:<10} {item['D_s_SW']:<10} {item['CIF_s']:<10} {item['CID_s']:<10}")

def imprimir_indices_sistema_completo(indices_sistema_multi):
    """Imprime los índices del sistema completo"""
    print("\n" + "="*120)
    print("ÍNDICES DE CONFIABILIDAD DEL SISTEMA COMPLETO")
    print("="*120)
    print(f"{'Parámetro':<30} {'Valor':<15} {'Unidad':<20}")
    print("-" * 70)
    print(f"{'Número de redes':<30} {indices_sistema_multi['total_redes']:<15} {'redes':<20}")
    print(f"{'Número de nodos':<30} {indices_sistema_multi['total_nodos']:<15} {'nodos':<20}")
    print(f"{'Número de clientes':<30} {indices_sistema_multi['total_clientes']:<15} {'clientes':<20}")
    print(f"{'SAIFI':<30} {indices_sistema_multi['SAIFI']:<15} {'interrupciones/año':<20}")
    print(f"{'SAIDI':<30} {indices_sistema_multi['SAIDI']:<15} {'h/año':<20}")
    print(f"{'ASAI':<30} {indices_sistema_multi['ASAI']:<15} {'%':<20}")
    print(f"{'EENS':<30} {indices_sistema_multi['EENS']:<15} {'MWh/año':<20}")

def imprimir_indices_por_red_separados(indices_redes_separadas, PSI_SS):
    """Imprime los índices por red separados"""
    print("\n" + "="*120)
    print("ÍNDICES POR RED (SEPARADOS)")
    print("="*120)
    
    for ss in PSI_SS:
        indices_red = indices_redes_separadas[ss]
        
        print(f"\n{'='*50}")
        print(f"RED {ss}")
        print(f"{'='*50}")
        print(f"{'Parámetro':<25} {'Valor':<15} {'Unidad':<15}")
        print("-" * 55)
        print(f"{'Transformadores Dist.':<25} {indices_red['transformadores_dist']:<15} {'unidades':<15}")
        print(f"{'Clientes':<25} {indices_red['clientes']:<15} {'clientes':<15}")
        print(f"{'SAIFI':<25} {indices_red['SAIFI']:<15} {'int/año':<15}")
        print(f"{'SAIDI':<25} {indices_red['SAIDI']:<15} {'h/año':<15}")
        print(f"{'ASAI':<25} {indices_red['ASAI']:<15} {'%':<15}")
        print(f"{'EENS':<25} {indices_red['EENS']:<15} {'MWh/año':<15}")

# ========================================
# PROGRAMA PRINCIPAL
# ========================================

if __name__ == "__main__":
    start_time = time.time()
    
    # EJECUTAR ANÁLISIS COMPLETO
    (PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES, indices_multi, 
     indices_sistema_multi, indices_redes_separadas) = ejecutar_analisis_completo()
    
    # MOSTRAR TODOS LOS RESULTADOS
    imprimir_informacion_redes(PSI_SS, TRANSFORMADORESPRINCIPALES_TRANSFORMADORES)
    imprimir_resultados_por_red(indices_multi, PSI_SS)
    imprimir_indices_sistema_completo(indices_sistema_multi)
    imprimir_indices_por_red_separados(indices_redes_separadas, PSI_SS)
    
    # Tiempo de ejecución
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTiempo total de resolución: {elapsed_time:.4f} segundos")