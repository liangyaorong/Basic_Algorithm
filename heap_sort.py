#coding:utf-8

def built_max_heap(vet):
    n = len(vet)
    while True:
        old = vet[:]
        for node_index in range(n):
            left_index = 2*node_index+1
            right_index = 2*node_index+2
            if left_index > n-1:     #若该结点无左孩，当然右孩也没有，则跳出循环
                break
            if right_index > n-1:    #若该结点有左孩，无右孩
                if vet[left_index]>vet[node_index]:
                    vet[left_index], vet[node_index] = vet[node_index], vet[left_index]
            if right_index <= n-1:   #若该节点既有左孩又有右孩
                if vet[left_index] >= vet[right_index] and vet[left_index] >= vet[node_index]:
                    vet[left_index], vet[node_index] = vet[node_index], vet[left_index]
                if vet[right_index] > vet[left_index] and vet[right_index] > vet[node_index]:
                    vet[right_index], vet[node_index] = vet[node_index], vet[right_index]
        if vet == old:
            break
    return vet

def get_max_and_new_heap(vet):
    vet[0], vet[-1] = vet[-1], vet[0]
    max = vet.pop()
    return max,vet

def heap_sort(vet):
    sorted_vet = []
    while len(vet)>=1:
        vet = built_max_heap(vet)
        max, new_heap = get_max_and_new_heap(vet)
        vet = new_heap
        sorted_vet.append(max)
    return sorted_vet

if __name__=='__main__':
    vet = [4, 4, 2, 5, 3, 9, 0, 6, 28, 9, -5, 7, 8]
    print heap_sort(vet)