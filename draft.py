import numpy as np

class ComparisonUnit:
    def __init__(self, condition, activation=1):
        self.condition = condition
        self.activation = activation

    def forward(self, x):
        return [self.activation if self.condition(val) else 0 for val in x]


class RelationUnit:
    def __init__(self, activation=1, threshold=0.1):
        self.activation = activation
        self.threshold = threshold

    def forward(self, x1, x2):
        return [self.activation if np.abs(val1 - val2) <= self.threshold else 0 for val1, val2 in zip(x1, x2)]


class SelectionCascade:
    def __init__(self, comparison_conditions, num_relation_units, aggregation_func=np.mean):
        self.comparison_units = [ComparisonUnit(cond) for cond in comparison_conditions]
        self.relation_units = [RelationUnit() for _ in range(num_relation_units)]
        self.aggregation_func = aggregation_func

    def forward(self, input_array):
        comparison_outputs = np.array([unit.forward(input_array) for unit in self.comparison_units])

        relation_outputs = []
        for i in range(0, len(comparison_outputs) - 1, 2):
            relation_outputs.append(self.relation_units[i // 2].forward(comparison_outputs[i], comparison_outputs[i + 1]))

        aggregated_output = self.aggregation_func(relation_outputs)
        return aggregated_output

# Example usage:
comparison_conditions = [
    lambda x: x > 5,
    lambda x: x < 12,
    lambda x: x % 2 == 0,
    lambda x: x % 3 == 0
]

input_data = np.array([7, 10, 6, 9])

selection_cascade = SelectionCascade(comparison_conditions, num_relation_units=2)
output = selection_cascade.forward(input_data)

print(output)
