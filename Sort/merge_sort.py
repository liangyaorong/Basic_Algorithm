# coding:utf-8

def merge(vet1, vet2):
    merge_vet = []
    index1 = 0
    index2 = 0
    while index1<len(vet1) and index2<len(vet2): # 将较小的放进merge_vet
        if vet1[index1]<vet2[index2]:
            merge_vet.append(vet1[index1])
            index1 += 1
        else:
            merge_vet.append(vet2[index2])
            index2 += 1
    if index1 == len(vet1):  # 若有一向量已全部放入merge_vet, 将剩下的部分全部复制进去(剩下部分之前已排好)
        merge_vet.extend(vet2[index2:])
    else:
        merge_vet.extend(vet1[index1:])
    return merge_vet

def merge_sort(vet):
    n = len(vet)
    if n <= 1:
        return vet
    middle_index = n/2
    left = merge_sort(vet[:middle_index])
    right = merge_sort(vet[middle_index:])
    return merge(left, right)

if __name__ == '__main__':
    vet = [4, 4, 2, 5, 3, 9, 0, 6, 28, 9, -5, 7, 8]
    print merge_sort(vet)
