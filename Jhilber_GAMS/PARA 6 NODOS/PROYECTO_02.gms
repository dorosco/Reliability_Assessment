$TITLE Sistema de Confiabilidad Generalizado para N Nodos - VERSIÓN FINAL CORREGIDA
$ONTEXT
================================================================================
PROGRAMA DE CONFIABILIDAD DE SISTEMAS ELÉCTRICOS RADIALES
Versión Final - Implementación exacta
================================================================================
$OFFTEXT
*===============================================================================
* PARTE 1: DECLARACIÓN DE CONJUNTOS BASE
*===============================================================================
OPTION DECIMALS = 8;
SETS
    n            /n1*n100/
    DATA_n       /Node, NC, L_MW/
    l            /l1*l100/
    DATA_l       /u, v, km, lambda, Tao_RP, Tao_SW/
    i            /1*100/
;

ALIAS (i, j, s, r, ii, jj);

*===============================================================================
* PARTE 2: PARÁMETROS DE ENTRADA
*===============================================================================

PARAMETERS
    DATA_NODOS(n,DATA_n)
    DATA_LINEAS(l,DATA_l)
;

*===============================================================================
* PARTE 3: IMPORTACIÓN DE DATOS
*===============================================================================

$ONECHO > gdxxrw.txt
par=DATA_NODOS rng=DATA_NODOS! rdim=1 cdim=1
$OFFECHO

$call gdxxrw DATA_NODOS.xlsx @gdxxrw.txt
$gdxIn DATA_NODOS.gdx
$load DATA_NODOS
$gdxIn

$ONECHO > gdxxrw2.txt
par=DATA_LINEAS rng=DATA_LINEAS! rdim=1 cdim=1
$OFFECHO

$call gdxxrw DATA_LINEAS.xlsx @gdxxrw2.txt
$gdxIn DATA_LINEAS.gdx
$load DATA_LINEAS
$gdxIn
DISPLAY DATA_LINEAS;
DISPLAY DATA_NODOS;
*===============================================================================
* PARTE 4: CONSTRUCCIÓN DE CONJUNTOS DINÁMICOS
*===============================================================================

SETS
    PSI_N(i)
    PSI_LN(i)
    PSI_SS(i)
    LINEAS(i,j)
    PSI_ADY(i,j)
;

PARAMETERS
    temp_LN(i)
    temp_u(i)
    temp_v(i)
    temp_linea(i,j)
;

* Inicialización
temp_LN(i) = 0;
temp_u(i) = 0;
temp_v(i) = 0;
temp_linea(i,j) = 0;

* Identificar nodos de carga
LOOP(n,
    LOOP(i,
        temp_LN(i)$(ABS(DATA_NODOS(n,'Node') - ORD(i)) < 0.5) = 1;
    );
);

PSI_LN(i)$(temp_LN(i) > 0.5) = YES;

* Identificar nodos en líneas
LOOP(l,
    LOOP(i,
        temp_u(i)$(ABS(DATA_LINEAS(l,'u') - ORD(i)) < 0.5) = 1;
        temp_v(i)$(ABS(DATA_LINEAS(l,'v') - ORD(i)) < 0.5) = 1;
    );
);

PSI_N(i)$((temp_u(i) + temp_v(i)) > 0.5) = YES;

PSI_SS(i)$(PSI_N(i) AND NOT PSI_LN(i)) = YES;

* Construir conjunto LINEAS (bidireccional)
LOOP(l,
    LOOP(i,
        LOOP(j,
            temp_linea(i,j)$((ABS(DATA_LINEAS(l,'u') - ORD(i)) < 0.5 AND 
                              ABS(DATA_LINEAS(l,'v') - ORD(j)) < 0.5) OR
                             (ABS(DATA_LINEAS(l,'u') - ORD(j)) < 0.5 AND 
                              ABS(DATA_LINEAS(l,'v') - ORD(i)) < 0.5)) = 1;
        );
    );
);

LINEAS(i,j)$(temp_linea(i,j) > 0.5) = YES;

* Conjunto de adyacencia PSI_ADY(s,j) = conjunto Psi_s del documento
PSI_ADY(s,j)$LINEAS(s,j) = YES;

DISPLAY PSI_N, PSI_LN, PSI_SS, LINEAS, PSI_ADY;

*===============================================================================
* PARTE 5: EXTRACCIÓN DE PARÁMETROS
*===============================================================================

PARAMETERS
    NC(i)
    L_MW(i)
    lambda_ij(i,j)
    tau_RP_ij(i,j)
    tau_SW_ij(i,j)
;

NC(i) = 0;
L_MW(i) = 0;
lambda_ij(i,j) = 0;
tau_RP_ij(i,j) = 0;
tau_SW_ij(i,j) = 0;

LOOP(n,
    LOOP(i$PSI_LN(i),
        NC(i)$(ABS(DATA_NODOS(n,'Node') - ORD(i)) < 0.5) = DATA_NODOS(n,'NC');
        L_MW(i)$(ABS(DATA_NODOS(n,'Node') - ORD(i)) < 0.5) = DATA_NODOS(n,'L_MW');
    );
);

LOOP(l,
    LOOP(i,
        LOOP(j$LINEAS(i,j),
            lambda_ij(i,j)$((ABS(DATA_LINEAS(l,'u') - ORD(i)) < 0.5 AND 
                             ABS(DATA_LINEAS(l,'v') - ORD(j)) < 0.5) OR
                            (ABS(DATA_LINEAS(l,'u') - ORD(j)) < 0.5 AND 
                             ABS(DATA_LINEAS(l,'v') - ORD(i)) < 0.5)) = 
                DATA_LINEAS(l,'lambda');
                
            tau_RP_ij(i,j)$((ABS(DATA_LINEAS(l,'u') - ORD(i)) < 0.5 AND 
                             ABS(DATA_LINEAS(l,'v') - ORD(j)) < 0.5) OR
                            (ABS(DATA_LINEAS(l,'u') - ORD(j)) < 0.5 AND 
                             ABS(DATA_LINEAS(l,'v') - ORD(i)) < 0.5)) = 
                DATA_LINEAS(l,'Tao_RP');
                
            tau_SW_ij(i,j)$((ABS(DATA_LINEAS(l,'u') - ORD(i)) < 0.5 AND 
                             ABS(DATA_LINEAS(l,'v') - ORD(j)) < 0.5) OR
                            (ABS(DATA_LINEAS(l,'u') - ORD(j)) < 0.5 AND 
                             ABS(DATA_LINEAS(l,'v') - ORD(i)) < 0.5)) = 
                DATA_LINEAS(l,'Tao_SW');
        );
    );
);

DISPLAY NC, L_MW, lambda_ij, tau_RP_ij, tau_SW_ij;

*===============================================================================
* PARTE 6: MODELO DE OPTIMIZACIÓN
*===============================================================================

PARAMETER nodo_objetivo;

VARIABLES
    f(i,j)
    g_SS(i)
    z_obj
;

POSITIVE VARIABLES f, g_SS;

EQUATIONS
    obj
    balance(i)
;

obj..
    z_obj =E= SUM(LINEAS(i,j), f(i,j) + f(j,i));

balance(i)$PSI_N(i)..
    SUM(j$LINEAS(i,j), f(i,j) - f(j,i))
    =E=
    g_SS(i)$PSI_SS(i) - (1$(ORD(i) EQ nodo_objetivo));

f.UP(i,j)$LINEAS(i,j) = 1;
g_SS.UP(i)$PSI_SS(i) = 1;

MODEL topo_caracterizacion /obj, balance/;

*===============================================================================
* PARTE 7: RESOLUCIÓN PARA TODOS LOS NODOS
*===============================================================================

PARAMETERS
    f_optimo(i,j,s)
    g_SS_optimo(i,s)
    objval(s)
;

f_optimo(i,j,s) = 0;
g_SS_optimo(i,s) = 0;
objval(s) = 0;

LOOP(s$PSI_LN(s),
    nodo_objetivo = ORD(s);
    SOLVE topo_caracterizacion USING LP MINIMIZING z_obj;
    LOOP(i,
        LOOP(j$LINEAS(i,j),
            f_optimo(i,j,s) = f.L(i,j);
        );
    );
    LOOP(i$PSI_SS(i),
        g_SS_optimo(i,s) = g_SS.L(i);
    );
    objval(s) = z_obj.L;
);

DISPLAY f_optimo, g_SS_optimo, objval;

*===============================================================================
* PARTE 8: CÁLCULO DE TASAS DE INTERRUPCIONES (CORREGIDO)
*===============================================================================

PARAMETERS
    N_RP(s)
    N_CB(i,j)
    N_T(s)
    N_SW(s)
    sum_inner_CB(s)
;

N_RP(s) = 0;
N_CB(i,j) = 0;
N_T(s) = 0;
N_SW(s) = 0;
sum_inner_CB(s) = 0;

* Ecuación (15): N_s^RP
* N_s^RP = sum_{(i,j) in Upsilon} (f_ijs + f_jis) * lambda_ij
* CORRECCIÓN: Usar $(ORD(i) < ORD(j)) para evitar contar dos veces la misma rama
* ya que LINEAS es bidireccional: LINEAS(i,j) y LINEAS(j,i) son la misma rama física
LOOP(s$PSI_LN(s),
    N_RP(s) = SUM(LINEAS(i,j)$(ORD(i) < ORD(j)), 
        (f_optimo(i,j,s) + f_optimo(j,i,s)) * lambda_ij(i,j));
);

DISPLAY "Ecuación (15) - N_RP calculado CORREGIDO", N_RP;

* Ecuación (19): N_{ij}^{CB}
* sum_{r in Psi_s} f_rss * lambda_rs
LOOP(s$PSI_LN(s),
    sum_inner_CB(s) = 0;
    LOOP(r$PSI_ADY(s,r),
        sum_inner_CB(s) = sum_inner_CB(s) + f_optimo(r,s,s) * lambda_ij(r,s);
    );
);

DISPLAY "Suma interna para N_CB", sum_inner_CB;

* N_{ij}^{CB} para interruptores en la subestación
LOOP(i$PSI_SS(i),
    LOOP(j$LINEAS(i,j),
        N_CB(i,j) = SUM(s$PSI_LN(s), sum_inner_CB(s) * f_optimo(i,j,s));
    );
);

DISPLAY "Ecuación (19) - N_CB calculado", N_CB;

* Ecuación (18): N_s^T
LOOP(s$PSI_LN(s),
    N_T(s) = 0;
    LOOP(i$PSI_SS(i),
        LOOP(j$LINEAS(i,j),
            N_T(s) = N_T(s) + N_CB(i,j) * (f_optimo(i,j,s) + f_optimo(j,i,s));
        );
    );
);

DISPLAY "Ecuación (18) - N_T calculado", N_T;

* Ecuación (17): N_s^SW
N_SW(s)$PSI_LN(s) = N_T(s) - N_RP(s);

DISPLAY "Ecuación (17) - N_SW calculado", N_SW;

*===============================================================================
* PARTE 9: CÁLCULO DE DURACIONES (CORREGIDO)
*===============================================================================

PARAMETERS
    D_RP(s)
    D_CB_SW(i,j)
    D_T_SW(s)
    D_bar_SW(s)
    D_SW(s)
    sum_inner_D_CB(s)
;

D_RP(s) = 0;
D_CB_SW(i,j) = 0;
D_T_SW(s) = 0;
D_bar_SW(s) = 0;
D_SW(s) = 0;
sum_inner_D_CB(s) = 0;

* Ecuación (25): D_s^RP
* D_s^RP = sum_{(i,j) in Upsilon} (f_ijs + f_jis) * lambda_ij * tau_RP_ij
* CORRECCIÓN: Usar $(ORD(i) < ORD(j)) para evitar contar dos veces
LOOP(s$PSI_LN(s),
    D_RP(s) = SUM(LINEAS(i,j)$(ORD(i) < ORD(j)), 
        (f_optimo(i,j,s) + f_optimo(j,i,s)) * lambda_ij(i,j) * tau_RP_ij(i,j));
);

DISPLAY "Ecuación (25) - D_RP calculado CORREGIDO", D_RP;

* Ecuación (30): D_{ij}^{CB,SW}
LOOP(s$PSI_LN(s),
    sum_inner_D_CB(s) = 0;
    LOOP(r$PSI_ADY(s,r),
        sum_inner_D_CB(s) = sum_inner_D_CB(s) + 
            f_optimo(r,s,s) * lambda_ij(r,s) * tau_SW_ij(r,s);
    );
);

DISPLAY "Suma interna para D_CB_SW", sum_inner_D_CB;

LOOP(i$PSI_SS(i),
    LOOP(j$LINEAS(i,j),
        D_CB_SW(i,j) = SUM(s$PSI_LN(s), sum_inner_D_CB(s) * f_optimo(i,j,s));
    );
);

DISPLAY "Ecuación (30) - D_CB_SW calculado", D_CB_SW;

* Ecuación (29): D_s^{T,SW}
LOOP(s$PSI_LN(s),
    D_T_SW(s) = 0;
    LOOP(i$PSI_SS(i),
        LOOP(j$LINEAS(i,j),
            D_T_SW(s) = D_T_SW(s) + D_CB_SW(i,j) * (f_optimo(i,j,s) + f_optimo(j,i,s));
        );
    );
);

DISPLAY "Ecuación (29) - D_T_SW calculado", D_T_SW;

* Ecuación (28): bar{D}_s^SW
* bar{D}_s^SW = sum_{(i,j) in Upsilon} (f_ijs + f_jis) * lambda_ij * tau_SW_ij
* CORRECCIÓN: Usar $(ORD(i) < ORD(j)) para evitar contar dos veces
LOOP(s$PSI_LN(s),
    D_bar_SW(s) = SUM(LINEAS(i,j)$(ORD(i) < ORD(j)), 
        (f_optimo(i,j,s) + f_optimo(j,i,s)) * lambda_ij(i,j) * tau_SW_ij(i,j));
);

DISPLAY "Ecuación (28) - D_bar_SW calculado CORREGIDO", D_bar_SW;

* Ecuación (27): D_s^SW
D_SW(s)$PSI_LN(s) = D_T_SW(s) - D_bar_SW(s);

DISPLAY "Ecuación (27) - D_SW calculado", D_SW;

*===============================================================================
* PARTE 10: ÍNDICES DE CONFIABILIDAD POR CLIENTE
*===============================================================================

PARAMETERS
    CIF(s)
    CID(s)
;

CIF(s)$PSI_LN(s) = N_RP(s) + N_SW(s);
CID(s)$PSI_LN(s) = D_RP(s) + D_SW(s);

DISPLAY "ÍNDICES POR CLIENTE:", CIF, CID;

*===============================================================================
* PARTE 11: ÍNDICES DEL SISTEMA
*===============================================================================

SCALARS
    total_clientes
    SAIFI
    SAIDI
    ASAI
    EENS
;

total_clientes = SUM(s$PSI_LN(s), NC(s));
SAIFI = SUM(s$PSI_LN(s), NC(s) * CIF(s)) / total_clientes;
SAIDI = SUM(s$PSI_LN(s), NC(s) * CID(s)) / total_clientes;
ASAI = 1 - (SAIDI / 8760);
EENS = SUM(s$PSI_LN(s), CID(s) * L_MW(s));

DISPLAY "=== ÍNDICES DEL SISTEMA ===";
DISPLAY total_clientes, SAIFI, SAIDI, ASAI, EENS;
