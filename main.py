import xml.etree.ElementTree as ET
import numpy as np

model = ET.parse("test_data_from_book.xml").getroot()
#  model = ET.parse("test.xml").getroot()
COMPARISON = 'comparison'
ALTERNATIVES = 'alternative'
WEIGHTS = 'weights'
CRITERION = 'criterion'
NAME1 = 'name1'
NAME2 = 'name2'
VALUE = 'value'

RI = {
    3: 0.5247,
    4: 0.8816,
    5: 1.1086,
    6: 1.2479,
    7: 1.3417,
    8: 1.4057,
    9: 1.4499,
    10: 1.4854
}


def comparisons(element):
    return element.findall(COMPARISON)


def comparisons_size(element):
    lines = float(len(comparisons(element)))
    result = 1
    while lines >= 1:
        result += 1
        lines /= result

    return result


def weights(element):
    return element.find(WEIGHTS)


def criteria(element):
    return element.findall(CRITERION)


def alternatives():
    return model.findall(ALTERNATIVES)


def alternatives_size():
    return len(alternatives())


def comparison_matrix(element):
    comparison_elements = comparisons(element)
    comp_mat = np.eye(comparisons_size(element))
    for comparison in comparison_elements:
        row = int(comparison.attrib[NAME1]) - 1
        col = int(comparison.attrib[NAME2]) - 1
        val = float(comparison.attrib[VALUE])

        comp_mat[row, col] = val
        comp_mat[col, row] = 1.0 / val

    return comp_mat


def priority_vector(comp_mat):
    eigen_values, eigen_vectors = np.linalg.eig(comp_mat)
    max_value = max(eigen_values)

    def find_vector():
        index = 0
        for ev in eigen_values:
            if ev == max_value:
                return eigen_vectors[:, index]
            else:
                index += 1

    result = find_vector()
    return result / np.linalg.norm(result, 1)


def weights_vector(criterion_element):
    w = weights(criterion_element)
    comp_mat = comparison_matrix(w)
    return priority_vector(comp_mat)


def has_criteria(element):
    return element.find(CRITERION) is not None


def final_ranking(parent_element=model):
    weights_vec = weights_vector(parent_element)
    criterion_vectors = []
    i = 0
    for criterion in criteria(parent_element):
        if has_criteria(criterion):
            criterion_vectors.append(
                final_ranking(criterion) * weights_vec[i]
            )
        else:
            criterion_vectors.append(
                priority_vector(comparison_matrix(criterion)) * weights_vec[i]
            )
        i += 1

    result = np.zeros(alternatives_size(), dtype='complex128')
    for vec in criterion_vectors:
        result += vec

    return result


def consistency_index(matrix):
    eigen_value = max(np.linalg.eigvals(matrix))
    n = matrix.shape[0]
    return float((eigen_value - n)) / (n - 1)


def random_consistency_index(matrix):
    return RI.get(matrix.shape[0], 0.1)


def consistency_ratio(matrix):
    return consistency_index(matrix) / random_consistency_index(matrix)


def is_consistent(matrix):
    return consistency_ratio(matrix) <= 0.10


def check_consistency(root=model.find(CRITERION), recursively=True):
    """True if consistent, False otherwise"""
    for child in criteria(root):
        if has_criteria(child) and recursively:
            con = check_consistency(child, recursively)
            if con is False:
                return False
        else:
            is_consistent(comparison_matrix(child))

    return True


if __name__ == "__main__":
    rank = final_ranking(model.find(CRITERION))
    print("Final ranking: ", rank)

    rank_sum = np.sum(rank)
    if rank_sum != 1.0:
        print("Something is not yes")
        print(rank_sum)

    i = 0
    for mark in rank:
        if mark == max(rank):
            for a in alternatives():
                if a.attrib['id'] == str(i):
                    print("The highest rated alternative is", a.attrib['name'])

        i += 1

    print("Consistency:", check_consistency())
