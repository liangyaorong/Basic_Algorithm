# coding:utf-8

def quick_sort_1(vet):
    if len(vet) <= 1:
        return vet

    key = vet[-1]
    middle = [i for i in vet if i==key]
    left = quick_sort_1([i for i in vet if i<key])
    right = quick_sort_1([i for i in vet if i>key])
    return left + middle + right

#-------------------------------------------------------------

def partition(vet, left_index, right_index):
    key = vet[left_index]
    while left_index < right_index:

        while left_index < right_index and vet[right_index] >= key:
            right_index -= 1
        if right_index != left_index:
            vet[right_index], vet[left_index] = vet[left_index], vet[right_index]
            left_index += 1

        while left_index < right_index and vet[left_index] < key:
            left_index += 1
        if right_index != left_index:
            vet[right_index], vet[left_index] = vet[left_index], vet[right_index]
            right_index -= 1
    return left_index


def quick_sort_2(vet, left_index, right_index):
    if left_index < right_index:
        key_index = partition(vet, left_index, right_index)
        quick_sort_2(vet, left_index, key_index-1)  # 原地排序，没有return
        quick_sort_2(vet, key_index+1, right_index)


if __name__ == '__main__':
    vet = [7,6,2,0,3,12,5,7,8,5,23,89,9,4,1,6,8]
    quick_sort_2(vet, 0, len(vet)-1)
    print vet

