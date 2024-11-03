# Generación y Análisis de Variables Aleatorias

Este proyecto explora diferentes métodos para generar variables aleatorias siguiendo una distribución normal estándar (media = 0, desviación estándar = 1.0) y comparar los resultados. Los métodos incluyen:

1. **Monty Python** (implementación en C)
2. **Box-Muller**
3. **GRAND** (Generador Random)
4. Generador estándar de `numpy` (`np.random.default_rng`)

## Estructura del Proyecto

- **Código fuente**: Implementaciones de los métodos de generación en Python
- **Análisis**: Scripts para comparar los métodos usando pruebas estadísticas como Kolmogorov-Smirnov y normalidad (`scipy.stats`).

## Instrucciones

1. Clonar el repositorio.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
# Normal-distribution-pseudo-random-generators
