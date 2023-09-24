import math

# Función para calcular la entropía de un conjunto de datos
def entropy(data):
    total_count = len(data)
    if total_count == 0:
        return 0

    # Contar las clases en el conjunto de datos
    class_counts = {}
    for item in data:
        label = item[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Calcular la entropía
    entropy_value = 0
    for label in class_counts:
        probability = class_counts[label] / total_count
        entropy_value -= probability * math.log2(probability)

    return entropy_value

# Función para calcular la ganancia de información
def information_gain(data, attribute_index):
    total_entropy = entropy(data)
    total_count = len(data)
    attribute_values = set([item[attribute_index] for item in data])
    weighted_entropy = 0

    for value in attribute_values:
        subset = [item for item in data if item[attribute_index] == value]
        subset_entropy = entropy(subset)
        subset_count = len(subset)
        weighted_entropy += (subset_count / total_count) * subset_entropy

    return total_entropy - weighted_entropy

# Función para construir el árbol de decisiones utilizando el algoritmo ID3
def build_decision_tree(data, attributes):
    # Caso base: si todos los ejemplos tienen la misma clase, retornar esa clase
    classes = set([item[-1] for item in data])
    if len(classes) == 1:
        return classes.pop()

    # Caso base: si no quedan atributos para dividir, retornar la clase mayoritaria
    if len(attributes) == 0:
        class_counts = {}
        for item in data:
            label = item[-1]
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        return max(class_counts, key=class_counts.get)

    # Seleccionar el mejor atributo para dividir
    best_attribute = max(attributes, key=lambda attr: information_gain(data, attr))

    # Crear un nodo de decisión para el mejor atributo
    decision_tree = {best_attribute: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]

    # Dividir los datos en función del mejor atributo
    attribute_values = set([item[best_attribute] for item in data])
    for value in attribute_values:
        subset = [item for item in data if item[best_attribute] == value]
        decision_tree[best_attribute][value] = build_decision_tree(subset, remaining_attributes)

    return decision_tree

# Datos de ejemplo (Horas de estudio, Asistencia a clases, Horas de sueño, Resultado)
data = [
    (10, 90, 7, 'Aprobar'),
    (5, 80, 6, 'No aprobar'),
    (12, 95, 8, 'Aprobar'),
    (8, 75, 6, 'No aprobar'),
    (6, 70, 5, 'No aprobar'),
    (15, 100, 7, 'Aprobar'),
    (9, 85, 6, 'Aprobar'),
    (4, 60, 5, 'No aprobar'),
    (7, 65, 6, 'No aprobar'),
    (11, 92, 7, 'Aprobar'),
]

# Atributos (Horas de estudio, Asistencia a clases, Horas de sueño)
attributes = [0, 1, 2]

# Construir el árbol de decisiones
decision_tree = build_decision_tree(data, attributes)

# Imprimir el árbol de decisiones
import pprint
pprint.pprint("----ADAN CLINTON VIZCARRA CHOQUE----")
pprint.pprint(decision_tree)
