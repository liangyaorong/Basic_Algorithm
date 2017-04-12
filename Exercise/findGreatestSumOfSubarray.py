def findGreatestSumOfSubarray(array):
    cursum = array[0]
    max_sum = -99999
    if len(array)<1:
        return 0

    for i in range(1,len(array)):
        if cursum+array[i]<array[i]:
            cursum = array[i]
        else:
            cursum = cursum+array[i]

        if cursum>max_sum:
            max_sum = cursum

    return max_sum

if __name__ == '__main__':
    list = [1,-2,3,10,-4,7,2,-5]
    print findGreatestSumOfSubarray(list)



